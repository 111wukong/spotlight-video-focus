"""MLX-SAM2 封装：在 M 系列 GPU 上做「点/框提示 → 全视频目标分割」。

为什么不用官方的 ``init_state(video_path)``
------------------------------------------
原实现会把整段视频解码成 ``float32`` 的 1024x1024 张量一次性堆在内存里
（1080p 的 250 帧约 3 GB，长视频直接爆），并且每帧的 ``maskmem_features``
也会永久累积（约 17 MB/帧）。这里做了两处改造：

1. **帧放磁盘**：``FrameStore`` 用 memmap 提供 ``uint8 RGB`` 帧，
   ``_get_image_feature`` 现场做预处理，内存与视频长度无关。
2. **状态裁剪**：``condition_with_memories`` 实际只会用到最近
   ``num_maskmem``（7）帧的掩码记忆和最近 ``max_obj_ptrs_in_encoder``（16）个
   object pointer，更早的非条件帧不会被读取。因此可以安全地把
   ``non_cond_frame_outputs`` 裁到最近 N 帧，既保住原生的时序连续性，
   又不会让内存随帧数线性增长。
"""

from __future__ import annotations

import time
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterator

import cv2
import mlx.core as mx
import numpy as np
from mlx_sam import SAM2VideoPredictor

from .video import FrameStore

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_u8(rgb: np.ndarray, size: int) -> np.ndarray:
    """``uint8 HWC RGB`` → ``float32 1CHW``，ImageNet 归一化（与 mlx-sam 对齐）。"""
    src = np.ascontiguousarray(rgb)
    h, w = src.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    x = cv2.resize(src, (size, size), interpolation=interp)
    x = x.astype(np.float32) * (1.0 / 255.0)
    x -= IMAGENET_MEAN
    x /= IMAGENET_STD
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])


def logits_to_soft(logits: np.ndarray) -> np.ndarray:
    """logits → 8bit 软掩码。保留边界处的过渡值，放大后边缘更自然。"""
    x = np.clip(logits, -30.0, 30.0)
    soft = 1.0 / (1.0 + np.exp(-x))
    return (soft * 255.0).astype(np.uint8)


class _MemPredictor(SAM2VideoPredictor):
    """内存受控的预测器：帧来自 FrameStore，状态可裁剪，并对外暴露掩码置信度。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_mask_iou = -1.0

    def _record_iou(self, out: dict) -> dict:
        """记下 SAM2 自报的掩码质量。

        基类的 ``_run_single_frame_inference`` 没有把 ``ious`` 放进输出字典，
        所以这里从两个解码入口把分数截下来。它比「面积突变」之类的启发式
        更能反映掩码到底靠不靠谱。
        """
        try:
            self.last_mask_iou = float(np.asarray(out["ious"]).reshape(-1).max())
        except Exception:
            self.last_mask_iou = -1.0
        return out

    def _predict_initial(self, encoded, points, labels):
        return self._record_iou(super()._predict_initial(encoded, points, labels))

    def _predict_tracked(self, *args, **kwargs):
        return self._record_iou(super()._predict_tracked(*args, **kwargs))

    def build_state(self, store: FrameStore) -> dict:
        state = {
            "images": store,  # 不是 ndarray，而是 FrameStore（见 _get_image_feature）
            "num_frames": int(len(store)),
            "offload_video_to_cpu": False,
            "offload_state_to_cpu": False,
            "async_loading_frames": False,
            "keep_image_features": False,
            "video_height": store.height,
            "video_width": store.width,
            "cached_features": {},
            "constants": {},
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_tracked_per_obj": {},
        }
        # 触发一次编码，确保权重已实际加载
        self._get_image_feature(state, 0)
        return state

    def _get_image_feature(self, inference_state: dict, frame_idx: int) -> dict:
        cached = inference_state["cached_features"].get(frame_idx)
        if cached is not None:
            return cached
        store: FrameStore = inference_state["images"]
        pixels = preprocess_u8(store.frame(frame_idx), self.image_size)
        encoded = self.model.encode_image(mx.array(pixels))
        # 只保留当前帧特征，避免特征缓存随帧数增长
        inference_state["cached_features"] = {frame_idx: encoded}
        return encoded

    def clear_frame(self, inference_state: dict, frame_idx: int, obj_id: int = 1) -> None:
        """彻底摘掉某一帧的提示与输出，以便换一种提示重来。

        注意不能直接用基类的 ``clear_all_prompts_in_frame``：它只是把输出从
        ``cond`` 挪到 ``non_cond``，于是下一次下提示时 ``prev_low`` 仍然存在，
        会走 ``_predict_tracked`` 分支去读一批还没编码记忆的条件帧
        （``maskmem_features`` 是 None），直接崩掉。这里连 ``non_cond`` 一起清掉，
        让这一帧回到「从没被提示过」的状态。
        """
        obj_idx = inference_state["obj_id_to_idx"].get(int(obj_id))
        if obj_idx is None:
            return
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        out_dict = inference_state["output_dict_per_obj"][obj_idx]
        out_dict["cond_frame_outputs"].pop(frame_idx, None)
        out_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        inference_state["frames_tracked_per_obj"][obj_idx].pop(frame_idx, None)

    def encode_pending_memory(self, inference_state: dict, frame_idx: int, obj_id: int = 1) -> bool:
        """把「刚下过提示的那一帧」写进记忆库。

        ``add_new_points_or_box`` 出于与 Meta 官方实现对齐的考虑，下提示时
        不会顺手编码记忆（官方是在传播前的 preflight 里统一做的）。我们是在
        传播途中动态重锚定，已经过了 preflight，所以必须手动补这一步，
        否则新的锚点不会真正影响后续帧。
        """
        obj_idx = inference_state["obj_id_to_idx"].get(int(obj_id))
        if obj_idx is None:
            return False
        out_dict = inference_state["output_dict_per_obj"][obj_idx]
        out = out_dict["cond_frame_outputs"].get(frame_idx) or out_dict["non_cond_frame_outputs"].get(frame_idx)
        if out is None:
            return False
        self._ensure_memory_encoded(inference_state, obj_idx, frame_idx, out, is_mask_from_points=True)
        return True

    def purge_tracking_memory(self, inference_state: dict, obj_id: int = 1) -> None:
        """清掉非条件帧的记忆。锚点被污染太久时用来「重新开始」。"""
        obj_idx = inference_state["obj_id_to_idx"].get(int(obj_id))
        if obj_idx is None:
            return
        out_dict = inference_state["output_dict_per_obj"][obj_idx]
        out_dict["non_cond_frame_outputs"].clear()

    def prune_state(self, inference_state: dict, keep: int = 32) -> int:
        """裁掉不会被再次读取的历史帧输出，返回释放的条目数。"""
        if keep <= 0:
            return 0
        dropped = 0
        for obj_idx, out_dict in inference_state["output_dict_per_obj"].items():
            non_cond: dict = out_dict["non_cond_frame_outputs"]
            if len(non_cond) <= keep:
                continue
            for frame_idx in sorted(non_cond)[:-keep]:
                non_cond.pop(frame_idx, None)
                dropped += 1
            tracked: dict = inference_state["frames_tracked_per_obj"][obj_idx]
            for frame_idx in sorted(tracked):
                if frame_idx not in non_cond and frame_idx not in out_dict["cond_frame_outputs"]:
                    tracked.pop(frame_idx, None)
        return dropped


class Segmenter:
    """对外的分割器门面。"""

    #: 档位 → 权重位置 / 型号 / 推理分辨率 / 工作分辨率长边
    #: ``model_id`` 必须显式给出：HuggingFace 上 small 档的权重文件名是
    #: ``model.safetensors``，里面不含任何型号线索，mlx-sam 无法自行推断。
    #:
    #: 速度主要由 ``image_size`` 决定而不是模型大小——记忆库的交叉注意力
    #: 开销随 ``image_size`` 的四次方增长，所以缩小推理分辨率比换小模型划算得多。
    #: M5 实测：768 → 5.9 帧/秒，1024 → 2.9 帧/秒。
    PRESETS: dict[str, dict] = {
        "turbo": {
            "subdir": "sam2.1-hiera-tiny-mlx",
            "filename": "sam2.1_hiera_tiny_image_segmenter.safetensors",
            "model_id": "facebook/sam2.1-hiera-tiny",
            "repo": "avbiswas/sam2.1-hiera-tiny-mlx",
            "image_size": 640,
            "long_side": 854,
            "label": "极速",
        },
        "fast": {
            "subdir": "sam2.1-hiera-small-mlx",
            "filename": "model.safetensors",
            "model_id": "facebook/sam2.1-hiera-small",
            "repo": "avbiswas/sam2.1-hiera-small-mlx",
            "image_size": 768,
            "long_side": 960,
            "label": "速度优先",
        },
        "balanced": {
            "subdir": "sam2.1-hiera-small-mlx",
            "filename": "model.safetensors",
            "model_id": "facebook/sam2.1-hiera-small",
            "repo": "avbiswas/sam2.1-hiera-small-mlx",
            "image_size": 1024,
            "long_side": 1280,
            "label": "均衡",
        },
        "quality": {
            "subdir": "sam2.1-hiera-base-plus-mlx",
            "filename": "sam2.1_hiera_base_plus_image_segmenter.safetensors",
            "model_id": "facebook/sam2.1-hiera-base-plus",
            "repo": "avbiswas/sam2.1-hiera-base-plus-mlx",
            "image_size": 1024,
            "long_side": 1600,
            "label": "质量优先",
        },
    }

    def __init__(
        self,
        checkpoint: str | Path,
        image_size: int = 1024,
        model_id: str | None = None,
        memory_dtype: str | None = None,
        memory_attention_dtype: str | None = None,
        cache_limit_gb: float = 4.0,
    ) -> None:
        self.checkpoint = str(checkpoint)
        self.image_size = int(image_size)
        if self.image_size % 16:
            raise ValueError(f"image_size 必须是 16 的倍数，收到 {self.image_size}")
        if cache_limit_gb > 0:
            try:
                mx.set_cache_limit(int(cache_limit_gb * 1024**3))
            except AttributeError:  # pragma: no cover - 旧版 mlx
                pass
        self.predictor = _MemPredictor(
            checkpoint=self.checkpoint,
            model_id=model_id,
            image_size=self.image_size,
            memory_dtype=memory_dtype,
            memory_attention_dtype=memory_attention_dtype,
        )
        self.last_timings: list[float] = []

    # ---------------------------------------------------------------- 状态

    def new_state(self, store: FrameStore) -> dict:
        return self.predictor.build_state(store)

    # ---------------------------------------------------------------- 提示

    @property
    def last_mask_iou(self) -> float:
        """最近一次解码给出的掩码置信度。"""
        return float(self.predictor.last_mask_iou)

    def prompt(
        self,
        state: dict,
        frame_idx: int,
        points: np.ndarray | None = None,
        labels: np.ndarray | None = None,
        box: tuple[float, float, float, float] | np.ndarray | None = None,
        obj_id: int = 1,
        clear_old_points: bool = True,
    ) -> np.ndarray:
        """在某一帧上给点/框提示，返回该帧的 logits (h, w)。

        ``points`` 是工作分辨率下的 ``(x, y)``；``box`` 是 ``(x1, y1, x2, y2)``。
        """
        box_arg = None
        if box is not None:
            box_arg = np.asarray(box, dtype=np.float32).reshape(2, 2)
        _, _, masks = self.predictor.add_new_points_or_box(
            state,
            frame_idx=int(frame_idx),
            obj_id=int(obj_id),
            points=None if points is None else np.asarray(points, dtype=np.float32).reshape(-1, 2),
            labels=None if labels is None else np.asarray(labels, dtype=np.int32).reshape(-1),
            clear_old_points=clear_old_points,
            box=box_arg,
        )
        return masks[0, 0]

    def prompt_mask(self, state: dict, frame_idx: int, mask_u8: np.ndarray, obj_id: int = 1) -> np.ndarray:
        """用一张掩码（0/255）重新锚定目标。"""
        m = (np.asarray(mask_u8, dtype=np.float32) / 255.0).astype(np.float32)
        _, _, masks = self.predictor.add_new_mask(state, int(frame_idx), int(obj_id), m)
        return masks[0, 0]

    def encode_pending_memory(self, state: dict, frame_idx: int, obj_id: int = 1) -> bool:
        """把刚下过提示的那一帧补进记忆库（传播途中动态重锚定必须调用）。"""
        return self.predictor.encode_pending_memory(state, int(frame_idx), int(obj_id))

    def clear_frame(self, state: dict, frame_idx: int, obj_id: int = 1) -> None:
        """清掉某一帧的提示，便于用不同提示重来（换提示前必须调用）。"""
        self.predictor.clear_frame(state, int(frame_idx), int(obj_id))

    def purge_tracking_memory(self, state: dict, obj_id: int = 1) -> None:
        """清空非条件帧的记忆，用于目标长时间丢失后「重新开始」。"""
        self.predictor.purge_tracking_memory(state, int(obj_id))

    # ---------------------------------------------------------------- 传播

    def track(
        self,
        state: dict,
        start_frame_idx: int,
        reverse: bool = False,
        frame_step: int = 1,
        keep_memory: int = 32,
        max_frames: int | None = None,
        on_frame: Callable[[int, np.ndarray, float], None] | None = None,
    ) -> Iterator[tuple[int, np.ndarray, float]]:
        """从 ``start_frame_idx`` 出发传播，逐帧产出 ``(frame_idx, logits, 掩码置信度)``。"""
        self.last_timings = []
        for frame_idx, _obj_ids, masks in self.predictor.propagate_in_video(
            state,
            start_frame_idx=int(start_frame_idx),
            max_frame_num_to_track=max_frames,
            reverse=reverse,
            return_masks=True,
            frame_step=int(frame_step),
        ):
            t0 = time.perf_counter()
            logits = masks[0, 0]
            iou = float(self.predictor.last_mask_iou)
            if on_frame is not None:
                on_frame(frame_idx, logits, iou)
            self.predictor.prune_state(state, keep=keep_memory)
            self.last_timings.append(time.perf_counter() - t0)
            yield frame_idx, logits, iou


def resolve_checkpoint(models_dir: str | Path, preset: str) -> tuple[str, int, int, str]:
    """按档位解析权重，返回 ``(checkpoint, image_size, work_long_side, model_id)``。"""
    models_dir = Path(models_dir)
    if preset not in Segmenter.PRESETS:
        raise KeyError(f"未知档位 {preset!r}，可选：{sorted(Segmenter.PRESETS)}")
    spec = Segmenter.PRESETS[preset]
    filename = spec["filename"]

    candidates = [models_dir / spec["subdir"] / filename]
    if models_dir.is_dir():
        candidates += list(models_dir.glob(f"*{filename}"))
    for c in candidates:
        if c.is_file():
            return str(c), int(spec["image_size"]), int(spec["long_side"]), str(spec["model_id"])
    raise FileNotFoundError(
        f"找不到 {preset} 档的权重（期望 {models_dir / spec['subdir'] / filename}）。"
        f"请运行 `python -m spotlight.cli download --preset {preset}` 下载。"
    )


_SEGMENTER_CACHE: dict[tuple[str, int], Segmenter] = {}


def get_segmenter(
    models_dir: str | Path, preset: str, work_long_side: int | None = None
) -> tuple[Segmenter, int]:
    """按档位取一个（带缓存的）分割器实例。

    加载权重需要 1~2 秒，网页端每次点选都重新加载会很难受，所以这里做进程内
    缓存。返回 ``(分割器, 工作分辨率长边)``。
    """
    checkpoint, image_size, default_long, model_id = resolve_checkpoint(models_dir, preset)
    key = (checkpoint, image_size)
    seg = _SEGMENTER_CACHE.get(key)
    if seg is None:
        seg = Segmenter(checkpoint, image_size=image_size, model_id=model_id)
        _SEGMENTER_CACHE[key] = seg
    return seg, (int(work_long_side) if work_long_side else default_long)


def clear_segmenter_cache() -> None:
    _SEGMENTER_CACHE.clear()
