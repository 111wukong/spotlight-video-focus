"""单目深度估计：Depth Anything V2 Small（ONNX，走 CoreML / ANE）。

用途
----
1. **真景深虚化**：目前背景是全局均匀高斯模糊，没有远近层次。有了逐像素深度，
   就能按「离焦面多远」分配模糊量——近处轻、远处重。
2. **深度辅助的背景重建**：单应矩阵只能对齐一个平面，走廊/街道这类有纵深的场景
   视差很重。深度可以把画面分成若干平面，各平面各自用一个单应矩阵。

深度语义（**很重要**）
----------------------
Depth Anything V2 输出的是**逆深度**（disparity 量纲）：**值越大 = 离镜头越近**。
本模块统一把它归一化成 ``0..1`` 的「近度」``near``（``1`` = 最近），
需要「距离」的地方用 ``1 - near``。写进深度视频的也是近度（``255`` = 最近）。

速度（M5 实测，CoreML EP）
--------------------------
518² 约 0.68 s/帧、378² 约 0.32 s、294² 约 0.20 s。
深度在时间上变化平缓，所以默认**抽帧 + 线性插值 + 时域平滑**，而不是逐帧推理。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

#: 权重来源（ONNX，含 CoreML 友好的 fp32 图）
DEPTH_REPO = "onnx-community/depth-anything-v2-small"
DEPTH_FILENAME = "onnx/model.onnx"

#: 推理输入边长，必须是 14 的倍数（模型内部按 14 分块）
DEFAULT_SIZE = 378

#: 一次最多缓存多少个采样帧的深度（控制内存）
MAX_SAMPLES = 400

#: 权重文件至少这么大才算下载完整（fp32 约 94 MB）
_MIN_BYTES = 50_000_000

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resolve_depth(models_dir: str | Path) -> Path | None:
    """找深度模型文件（ONNX）。"""
    cand = Path(models_dir) / "depth" / "model.onnx"
    if not cand.is_file() or cand.stat().st_size < _MIN_BYTES:
        return None
    return cand


class DepthEstimator:
    """Depth Anything V2 Small，ONNX Runtime + CoreML EP。

    接口很小：``estimate(rgb)`` 收一张 RGB 图，回一张同尺寸的**原始逆深度**图
    （float32，值越大越近，量纲是模型内部的，不要直接当 0..1 用）。
    归一化交给 :func:`normalize_depth`。
    """

    def __init__(
        self,
        model_path: str | Path,
        size: int = DEFAULT_SIZE,
        threads: int = 4,
    ) -> None:
        self.model_path = Path(model_path)
        self.available = False
        self.reason = ""
        self._session = None
        self._in_name = "pixel_values"
        self._out_name = "predicted_depth"
        self.size = max(14, int(size) // 14 * 14)
        self._input_size = self.size

        if not self.model_path.is_file():
            self.reason = f"模型文件不存在：{self.model_path}"
            return
        try:
            import onnxruntime as ort

            providers = [
                p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                if p in ort.get_available_providers()
            ]
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(threads)
            options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=providers
            )
            self._in_name = self._session.get_inputs()[0].name
            self._out_name = self._session.get_outputs()[0].name
            self.available = True
        except Exception as exc:  # noqa: BLE001 - 缺 onnxruntime / 权重损坏都算不可用
            self.reason = f"{type(exc).__name__}: {exc}"

    # ------------------------------------------------------------------ 单帧

    def estimate(self, rgb: np.ndarray) -> np.ndarray:
        """跑一帧，返回**原始**逆深度（float32，``size×size``，值越大越近）。"""
        if not self.available or self._session is None:
            raise RuntimeError(self.reason or "深度模型不可用")
        side = self._input_size
        interp = cv2.INTER_AREA if max(rgb.shape[:2]) > side else cv2.INTER_CUBIC
        img = cv2.resize(rgb, (side, side), interpolation=interp)
        x = img.astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        x = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])
        out = self._session.run([self._out_name], {self._in_name: x})[0]
        return np.squeeze(out).astype(np.float32)


def normalize_depth(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """把原始逆深度线性映射成 ``0..1`` 的近度（1 = 最近）。

    ``lo`` / ``hi`` 必须是**整段共用**的两个常数——逐帧各自归一化会让
    同样的景物在不同帧得到不同的近度，虚化强度就会闪。
    """
    span = max(1e-6, float(hi) - float(lo))
    return np.clip((raw.astype(np.float32) - float(lo)) / span, 0.0, 1.0)


def to_blur_level(
    near_u8: np.ndarray,
    focus_near: float = 0.6,
    near_scale: float = 0.35,
) -> np.ndarray:
    """把「近度」图转成「模糊层级」图（``uint8``，``0`` = 最清晰，``255`` = 最糊）。

    以**焦平面**（目标所在深度 ``focus_near``）为原点分两侧：

    * 比焦平面**远**（近度更小）→ 线性升到 ``1.0``，即远处最糊；
    * 比焦平面**近**（近度更大）→ 线性升到 ``near_scale``，即近景只是轻微失焦。

    这就是「近处轻、远处重」，同时保证目标所在的平面始终是清晰的。
    层级映射在这里（Python / numpy）算好，ffmpeg 那边只做阈值和斜坡，
    避免把复杂的逐像素表达式塞进 ``lut``。
    """
    f = float(np.clip(focus_near, 0.0, 1.0))
    n = np.asarray(near_u8, dtype=np.float32) / 255.0
    far_side = (f - n) / max(1e-3, f)             # >0 表示比焦平面远
    near_side = (n - f) / max(1e-3, 1.0 - f)      # >0 表示比焦平面近
    level = np.where(n <= f, far_side, near_side * float(near_scale))
    return (np.clip(level, 0.0, 1.0) * 255.0).astype(np.uint8)


# ---------------------------------------------------------------------- 整段


def estimate_clip(
    estimator: DepthEstimator,
    store,
    *,
    step: int = 3,
    smooth: float = 0.6,
    size: int | None = None,
    target_masks=None,
    on_progress: Callable[..., None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[float | None, Iterator[np.ndarray]]:
    """对整段视频估计深度。

    返回 ``(focus_near, frames)``：

    * ``focus_near``：目标所在位置的近度（``0..1``）。给了 ``target_masks`` 才有，
      否则是 ``None``。虚化时用它当「焦平面」——目标所在的深度保持清晰。
    * ``frames``：逐帧产出的近度图（``uint8``，``255`` = 最近），尺寸与 ``store`` 一致。
      这是一个**生成器**，调用方边取边写（配合 ``write_mask_video``）。

    抽帧 + 线性插值 + 时域平滑：深度在时间上变化平缓，逐帧推理纯属浪费。
    ``smooth`` 是时域 EMA 系数（0 = 不平滑，越大越稳但越滞后）。
    """
    if not estimator.available:
        raise RuntimeError(estimator.reason or "深度模型不可用")

    n = int(store.n_frames)
    if n <= 0:
        return None, iter(())

    # 控制内存：采样帧数封顶，必要时自动放大步长
    eff_step = max(1, int(step))
    if n / eff_step > MAX_SAMPLES:
        eff_step = int(np.ceil(n / MAX_SAMPLES))
    idxs = list(range(0, n, eff_step))
    if idxs[-1] != n - 1:
        idxs.append(n - 1)

    side = max(14, int(size or estimator.size) // 14 * 14)

    # ---- 第一遍：抽帧推理，缓存原始逆深度（float16，省一半内存）
    started = time.perf_counter()
    samples: list[np.ndarray] = []
    for k, i in enumerate(idxs):
        if should_cancel is not None and should_cancel():
            raise RuntimeError("已取消")
        raw = estimator.estimate(store.frame(i))
        if raw.shape[0] != side:
            raw = cv2.resize(raw, (side, side), interpolation=cv2.INTER_LINEAR)
        samples.append(raw.astype(np.float16))
        if on_progress is not None:
            on_progress("depth", k + 1, len(idxs), f"估计深度 {k + 1}/{len(idxs)}")

    # ---- 整段共用的归一化区间（用分位数，避免个别极值把动态范围吃掉）
    flat = np.concatenate([s.ravel() for s in samples])
    lo, hi = (float(v) for v in np.percentile(flat, [2.0, 98.0]))
    if hi - lo < 1e-3:                      # 近乎纯平面：给一个兜底区间
        lo, hi = float(flat.min()), float(flat.min()) + 1e-3
    del flat

    # ---- 目标所在深度（焦平面）
    focus_near: float | None = None
    if target_masks is not None:
        vals: list[float] = []
        for i, raw16 in zip(idxs, samples):
            if i >= target_masks.n:
                continue
            m = target_masks.get(i)
            if m.shape != (side, side):
                m = cv2.resize(m, (side, side), interpolation=cv2.INTER_NEAREST)
            sel = m > 127
            if int(sel.sum()) < 32:
                continue
            vals.append(float(np.median(raw16.astype(np.float32)[sel])))
        if vals:
            focus_near = float(normalize_depth(np.array(np.median(vals)), lo, hi))

    # ---- 第二遍：逐帧插值 + 时域平滑，产出近度图
    def frames() -> Iterator[np.ndarray]:
        prev: np.ndarray | None = None
        pos = 0
        for f in range(n):
            if should_cancel is not None and should_cancel():
                raise RuntimeError("已取消")
            while pos + 1 < len(idxs) and idxs[pos + 1] <= f:
                pos += 1
            i0 = idxs[pos]
            i1 = idxs[min(pos + 1, len(idxs) - 1)]
            d0 = samples[pos].astype(np.float32)
            if i1 == i0:
                cur = d0
            else:
                t = (f - i0) / float(i1 - i0)
                cur = d0 * (1.0 - t) + samples[min(pos + 1, len(samples) - 1)].astype(np.float32) * t
            near = normalize_depth(cur, lo, hi)
            if prev is not None and smooth > 0:
                near = prev * float(smooth) + near * (1.0 - float(smooth))
            prev = near
            if near.shape != (store.height, store.width):
                near = cv2.resize(near, (store.width, store.height), interpolation=cv2.INTER_LINEAR)
            yield (near * 255.0).astype(np.uint8)

    if on_progress is not None:
        on_progress(
            "depth", len(idxs), len(idxs),
            f"深度估计完成（{len(idxs)} 帧采样 / {time.perf_counter() - started:.0f} 秒）",
        )
    return focus_near, frames()
