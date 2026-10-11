"""端到端流水线：选人 → 全视频分割 → 目标守卫 → 合成出片。

内存约定
--------
整条链路的内存占用与视频长度**无关**：

* 帧存在磁盘 memmap 里（:class:`~spotlight.video.FrameStore`）；
* 掩码也存在磁盘 memmap 里（:class:`~spotlight.video.MaskStore`）；
* 补齐空洞、插值、算存在门都是流式的 O(1) 处理。

早期版本把每帧掩码留在内存里，1280x720 每帧 0.92 MB，300 秒素材就是 8.3 GB，
再被补齐和插值各复制一份 —— 长视频直接爆掉。
"""

from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .appearance import AppearanceEngine, mask_centroid
from .cleanplate import CleanConfig, CleanError, remove_people
from .composite import EffectParams, render, reencode_pair
from .segmenter import Segmenter, get_segmenter, logits_to_soft
from .tracker import GuardConfig, TargetGuard
from .video import (
    FrameStore,
    MaskStore,
    VideoInfo,
    gate_levels,
    lost_segments,
    probe,
    summarize_masks,
    write_mask_video,
)

ProgressFn = Callable[["Progress"], None]

#: 各档位在 M5 上的实测处理速度（帧/秒），用于给用户估算耗时。
PRESET_SPEED: dict[str, float] = {
    "turbo": 9.5,
    "fast": 5.8,
    "balanced": 2.9,
    "quality": 1.6,
}


class Cancelled(RuntimeError):
    """用户主动中断。"""


@dataclass
class Selection:
    """用户在某一帧上给目标下的提示。

    坐标一律用 **归一化** 的 ``0..1``（相对画面宽高），这样前端不需要知道
    工作分辨率，也不会因为预览缩放而对不上。
    """

    frame_idx: int
    points: list[tuple[float, float]] = field(default_factory=list)
    labels: list[int] = field(default_factory=list)
    box: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.points and self.box is None:
            raise ValueError("必须提供至少一个点或一个框")
        if self.labels and len(self.labels) != len(self.points):
            raise ValueError("labels 与 points 数量不一致")

    def label(self, idx: int) -> int:
        if idx < len(self.labels):
            return int(self.labels[idx])
        return 1

    def describe(self) -> str:
        bits = []
        if self.box:
            bits.append("框选")
        pos = sum(1 for i in range(len(self.points)) if self.label(i) == 1)
        neg = len(self.points) - pos
        if pos:
            bits.append(f"{pos} 个正样本点")
        if neg:
            bits.append(f"{neg} 个排除点")
        return f"第 {self.frame_idx} 帧 · " + "，".join(bits or ["—"])


@dataclass
class RunConfig:
    preset: str = "balanced"
    effects: EffectParams = field(default_factory=EffectParams)
    guard: GuardConfig = field(default_factory=GuardConfig)
    encoder: str = "auto"
    quality: int = 60
    work_long_side: int | None = None
    frame_step: int = 1
    max_frames: int | None = None
    frame_start: int | None = None
    frame_end: int | None = None
    keep_memory: int = 32
    audio: bool = True
    reid: bool = True
    speck_ratio: float = 0.08
    gate_ramp_up: int = 3
    gate_ramp_down: int = 7
    timeline_limit: int = 4000
    #: 是否把除目标外的所有人抹掉（用重建的背景补上）
    remove_others: bool = False
    clean: CleanConfig = field(default_factory=CleanConfig)
    #: 是否用深度做「真景深」虚化（近处轻、远处重）。需要 depth 权重。
    depth_blur: bool = False
    #: 深度估计的抽帧步长（越大越快，深度在时间上变化平缓）
    depth_step: int = 3


@dataclass
class Progress:
    stage: str
    done: int = 0
    total: int = 0
    message: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def fraction(self) -> float:
        return (self.done / self.total) if self.total else 0.0


@dataclass
class RunResult:
    output_path: str
    info: VideoInfo
    n_frames: int
    mask_size: tuple[int, int]
    elapsed: float
    frames_per_second: float
    guard_summary: dict
    mask_stats: dict
    lost_segments: list[dict] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)
    clean_stats: dict = field(default_factory=dict)


def estimate_seconds(n_frames: int, preset: str, frame_step: int = 1) -> float:
    """粗估处理耗时（秒）。速度数据来自 M5 实测，用来给界面一个数量级的感觉。"""
    speed = PRESET_SPEED.get(preset, 2.9) / max(1, int(frame_step))
    # 推理之外还有解码、合成、编码的固定开销，按每帧 25% 估
    return n_frames / max(0.1, speed) * 1.25


def _emit(cb: ProgressFn | None, stage: str, done: int = 0, total: int = 0, message: str = "", **extra) -> None:
    if cb is not None:
        cb(Progress(stage=stage, done=done, total=total, message=message, extra=extra))


def _clean_specks(soft: np.ndarray, min_ratio: float) -> np.ndarray:
    """丢掉掩码里的零散小碎块，只保留主要连通域。

    SAM2 偶尔会在画面别处留下几个像素的杂散响应，放大后就是一小块莫名其妙
    的清晰区域，这里统一清掉。

    连通域分析放到 1/4 尺度上做：分辨率降 16 倍，开销也降一个数量级，
    而「找出哪些小块该丢」这件事本来就不需要全分辨率。
    """
    if min_ratio <= 0:
        return soft
    h, w = soft.shape[:2]
    if h < 32 or w < 32:
        return soft
    small = cv2.resize(soft, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
    binary = (small > 100).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 2:
        return soft
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return soft
    keep_threshold = max(1.0, float(areas.max()) * min_ratio)
    keep_labels = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= keep_threshold]
    if len(keep_labels) == n - 1:
        return soft
    keep_small = np.isin(labels, keep_labels).astype(np.uint8)
    keep_mask = cv2.resize(keep_small, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.where(keep_mask > 0, soft, 0).astype(np.uint8)


def _px_points(sel: Selection, w: int, h: int) -> np.ndarray | None:
    if not sel.points:
        return None
    return np.asarray([[px * w, py * h] for px, py in sel.points], dtype=np.float32)


def _px_box(sel: Selection, w: int, h: int) -> np.ndarray | None:
    if sel.box is None:
        return None
    x1, y1, x2, y2 = sel.box
    return np.asarray([x1 * w, y1 * h, x2 * w, y2 * h], dtype=np.float32)


def _coverage(logits: np.ndarray, box: np.ndarray) -> float:
    """掩码在给定框内的覆盖比例。"""
    h, w = logits.shape[:2]
    x1 = max(0, int(round(box[0])))
    y1 = max(0, int(round(box[1])))
    x2 = min(w, int(round(box[2])))
    y2 = min(h, int(round(box[3])))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return float((logits[y1:y2, x1:x2] > 0.0).mean())


def _find_reference_box(
    people: list,
    px_points: np.ndarray | None,
    base_logits: np.ndarray,
) -> np.ndarray | None:
    """找出「用户想框住的那个人」的大致范围。

    优先用包含点击位置的人体框；如果检测恰好漏掉了那个人（在密集画面里很常见），
    就退一步用**掩码自身的质心**去找——SAM2 只切出头部时，那个头显然还在人的
    身体框里，用这个框就能把提示扩展到全身。
    """
    if not people:
        return None
    hits = []
    if px_points is not None and len(px_points):
        hits = [p for p in people if p.x1 <= px_points[0][0] <= p.x2 and p.y1 <= px_points[0][1] <= p.y2]
    if not hits:
        centroid = mask_centroid((base_logits > 0).astype(np.uint8) * 255)
        if centroid is not None:
            hits = [p for p in people if p.x1 <= centroid[0] <= p.x2 and p.y1 <= centroid[1] <= p.y2]
    if not hits:
        return None
    return np.asarray(min(hits, key=lambda p: p.area).as_tuple(), dtype=np.float32)


def resolve_prompt(
    seg: Segmenter,
    state: dict,
    engine: AppearanceEngine,
    store: FrameStore,
    selection: Selection,
) -> tuple[np.ndarray, str]:
    """把用户给的点/框变成「最能覆盖整个人」的 SAM2 提示。

    为什么需要这一步：在拥挤画面里直接给一个点，SAM2 很容易只把**头部**切出来
    （头是个更「干净」的对象），结果就是只有脸是清晰的、身体被虚化。
    这里先用人体检测估出目标的大致范围，如果掩码没填满这个范围，就换成
    「人体框」再下一次提示，取覆盖更好的那个。

    返回 ``(logits, 说明)``。
    """
    w, h = store.width, store.height
    px_points = _px_points(selection, w, h)
    px_labels = np.asarray(selection.labels, dtype=np.int32) if selection.labels else None
    px_box = _px_box(selection, w, h)
    idx = max(0, min(store.n_frames - 1, int(selection.frame_idx)))

    base = seg.prompt(state, idx, points=px_points, labels=px_labels, box=px_box)

    ref = px_box
    if ref is None:
        try:
            people = engine.detect_people(store.frame(idx))
        except Exception:
            people = []
        ref = _find_reference_box(people, px_points, base)

    if ref is None:
        return base, "没找到可参考的人体框，按你的选择直接分割"

    cover = _coverage(base, ref)
    if cover >= 0.45:
        return base, f"掩码已覆盖目标区域 {cover:.0%}"

    seg.clear_frame(state, idx)
    # 只用人体框、不带用户的点：点落在头部时会把 SAM2 又拉回头部去
    alt = seg.prompt(state, idx, box=ref, clear_old_points=True)
    cover_alt = _coverage(alt, ref)
    if cover_alt > cover + 0.05:
        return alt, f"已自动扩展到整个人（覆盖 {cover:.0%} → {cover_alt:.0%}）"
    seg.clear_frame(state, idx)
    return seg.prompt(state, idx, points=px_points, labels=px_labels, box=px_box), (
        f"按你的选择分割（覆盖 {cover:.0%}，扩展无改善）"
    )


def preview_mask(
    video_path: str | Path,
    selection: Selection,
    models_dir: str | Path,
    preset: str = "balanced",
    work_long_side: int | None = None,
    tmp_dir: str | Path | None = None,
    store: FrameStore | None = None,
) -> tuple[np.ndarray, tuple[int, int], VideoInfo, str]:
    """只处理用户选中的那一帧，返回 ``(软掩码, 工作分辨率, 视频信息, 说明)``。"""
    info = probe(video_path)
    seg, long_side = get_segmenter(models_dir, preset, work_long_side)
    own_work = tmp_dir is None and store is None
    work = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="spotlight-preview-"))
    try:
        if store is None:
            store = FrameStore(video_path, work, long_side=long_side, info=info)
        state = seg.new_state(store)
        logits, note = resolve_prompt(seg, state, AppearanceEngine(), store, selection)
        return logits_to_soft(logits), (store.width, store.height), info, note
    finally:
        if own_work:
            shutil.rmtree(work, ignore_errors=True)


def _normalize_selections(selections) -> list[Selection]:
    if isinstance(selections, Selection):
        return [selections]
    items = [s for s in selections if s is not None]
    if not items:
        raise ValueError("至少需要一个选人提示")
    return sorted(items, key=lambda s: int(s.frame_idx))


def _write_depth_video(
    cfg: RunConfig,
    models_dir: str | Path,
    store: FrameStore,
    masks: MaskStore,
    work: Path,
    on_progress: ProgressFn | None,
    should_cancel: Callable[[], bool] | None,
) -> Path | None:
    """估计深度，写成「模糊层级」灰度视频，供 ffmpeg 做分层景深虚化。

    返回 ``None`` 表示走不了景深（没下权重 / 模型不可用），调用方会回退到
    原来的全局均匀虚化——景深是可选增强，不该因为它让整条流程失败。
    """
    from .depth import DepthEstimator, estimate_clip, resolve_depth, to_blur_level

    model = resolve_depth(models_dir)
    if model is None:
        _emit(on_progress, "depth", 0, 1,
              "未找到深度模型，回退到均匀虚化（可执行 spotlight download --preset depth）")
        return None
    estimator = DepthEstimator(model)
    if not estimator.available:
        _emit(on_progress, "depth", 0, 1,
              f"深度模型不可用（{estimator.reason}），回退到均匀虚化")
        return None

    focus_near, frames = estimate_clip(
        estimator, store,
        step=cfg.depth_step,
        target_masks=masks,
        on_progress=lambda *a, **kw: _emit(on_progress, *a, **kw),
        should_cancel=should_cancel,
    )
    # 焦平面取目标所在深度；估不出来时用一个「中景偏前」的保守默认值。
    focus = 0.6 if focus_near is None else float(focus_near)
    path = work / "depth.mkv"
    write_mask_video(
        (to_blur_level(f, focus) for f in frames),
        path, store.width, store.height, store.info.fps_expr, total=store.n_frames,
    )
    _emit(on_progress, "depth", 1, 1,
          f"景深层级已生成（焦平面近度 {focus:.2f}）")
    return path


def run(
    video_path: str | Path,
    selections: Selection | Sequence[Selection],
    out_path: str | Path,
    models_dir: str | Path,
    config: RunConfig | None = None,
    on_progress: ProgressFn | None = None,
    tmp_dir: str | Path | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> RunResult:
    """跑完整流程。

    ``selections`` 可以给多个：第一个是初始选人，其余是用户在别的帧上补的修正点
    （跟踪跑偏后手动接管用）。它们都会作为条件帧注入，SAM2 传播到那一帧时会直接采用。
    """
    cfg = config or RunConfig()
    anchors = _normalize_selections(selections)
    anchor_frames = {int(a.frame_idx) for a in anchors}
    t_start = time.perf_counter()
    info = probe(video_path)

    seg, long_side = get_segmenter(models_dir, cfg.preset, cfg.work_long_side)

    work = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(prefix="spotlight-run-"))
    work.mkdir(parents=True, exist_ok=True)
    own_work = tmp_dir is None

    def check_cancel() -> None:
        if should_cancel is not None and should_cancel():
            raise Cancelled("已取消")

    try:
        _emit(on_progress, "decode", 0, 1, f"解码视频（工作分辨率长边 {long_side}）")
        store = FrameStore(video_path, work, long_side=long_side, info=info)
        n = store.n_frames
        lo = max(0, int(cfg.frame_start or 0))
        hi = n if cfg.frame_end is None else min(n, int(cfg.frame_end))
        if cfg.max_frames:
            hi = min(hi, lo + int(cfg.max_frames))
        if hi - lo < 2:
            raise RuntimeError("可处理的帧太少，请检查帧范围设置")
        _emit(on_progress, "decode", 1, 1, f"共 {n} 帧，{store.width}x{store.height}，本次处理第 {lo}~{hi - 1} 帧")
        check_cancel()

        _emit(on_progress, "load", 0, 1, f"加载 SAM2 权重（{cfg.preset}）")
        state = seg.new_state(store)
        engine = AppearanceEngine()
        guard = TargetGuard(engine, cfg.guard)

        # ---------------------------------------------------------- 注入锚点
        masks = MaskStore(work / "masks.u8", n, store.width, store.height)
        primary_logits = None
        for order, anchor in enumerate(anchors):
            k = max(lo, min(hi - 1, int(anchor.frame_idx)))
            logits, note = resolve_prompt(seg, state, engine, store, anchor)
            soft = logits_to_soft(logits)
            if int((soft > 100).sum()) < 32:
                raise RuntimeError(
                    f"在第 {k} 帧的指定位置没有分割出有效目标，请重新选择"
                    "（尽量点在人物身体上）"
                )
            masks.set(k, _clean_specks(soft, cfg.speck_ratio))
            if order == 0:
                primary_logits = logits
                guard.seed(store.frame(k), logits)
            else:
                guard.accept_manual(k, store.frame(k), logits)
            if note:
                _emit(on_progress, "load", 1, 1, f"锚点 {order + 1}：{note}")
            check_cancel()

        _emit(on_progress, "load", 1, 1, "权重与锚点就绪")
        k0 = max(lo, min(hi - 1, int(anchors[0].frame_idx)))
        total_steps = (hi - lo) + len(anchors)
        done = masks.n_written

        # ---------------------------------------------------------- 传播
        def handle(frame_idx: int, logits: np.ndarray, mask_iou: float) -> None:
            nonlocal done
            if frame_idx < lo or frame_idx >= hi:
                return
            check_cancel()
            rgb = store.frame(frame_idx)
            result = guard.observe(frame_idx, rgb, logits, mask_iou)
            if result.purge_memory:
                seg.purge_tracking_memory(state, obj_id=1)
            if result.reanchor_box is not None:
                box = np.asarray(result.reanchor_box.as_tuple(), dtype=np.float32)
                # 先下新提示，再把这一帧补进记忆库——否则新锚点不会影响后续帧
                fresh = seg.prompt(state, frame_idx, box=box, obj_id=1)
                seg.encode_pending_memory(state, frame_idx, obj_id=1)
                result.soft_mask = logits_to_soft(fresh)
            # 用户手动指定的帧，其掩码在注入阶段就写好了，守卫不能把它覆盖掉
            if frame_idx not in anchor_frames:
                masks.set(frame_idx, _clean_specks(result.soft_mask, cfg.speck_ratio))
            done += 1
            _emit(
                on_progress, "track", done, total_steps,
                f"跟踪中 {min(done, total_steps)}/{total_steps}",
                state=result.state.value, reanchors=guard.reanchors, lost=guard.total_lost,
            )

        if k0 < hi - 1:
            _emit(on_progress, "track", done, total_steps, "向后跟踪")
            for _ in seg.track(
                state, start_frame_idx=k0, reverse=False, frame_step=cfg.frame_step,
                keep_memory=cfg.keep_memory, max_frames=hi - k0, on_frame=handle,
            ):
                pass

        if k0 > lo:
            _emit(on_progress, "track", done, total_steps, "向前跟踪（回到片头）")
            for _ in seg.track(
                state, start_frame_idx=k0 - 1, reverse=True, frame_step=cfg.frame_step,
                keep_memory=cfg.keep_memory, max_frames=k0 - lo, on_frame=handle,
            ):
                pass

        check_cancel()

        # ---------------------------------------------------------- 深度（可选）
        # 必须在 store.close() 之前跑——深度估计要逐帧读原图。
        # 清除旁人模式整段保持清晰，不需要景深，所以那边不跑。
        depth_path: Path | None = None
        if cfg.depth_blur and not cfg.remove_others:
            depth_path = _write_depth_video(
                cfg, models_dir, store, masks, work, on_progress, should_cancel,
            )
            check_cancel()

        # 传播结束后帧就不再需要了，提前解除映射，把页缓存还给系统。
        # （帧是只读的 memmap，属于可回收的文件页、不是泄漏，但长视频下
        # 它的体量不小，早点放掉能让后面合成阶段的内存更宽裕。）
        store.close()

        # ---------------------------------------------------------- 清除旁人
        clean_stats: dict = {}
        if cfg.remove_others:
            _emit(on_progress, "clean", 0, 1, "清除旁人：正在分割画面里所有的人")
            cleaned = work / "cleaned.mkv"
            try:
                clean_stats = remove_people(
                    video_path, cleaned, masks, models_dir, work / "clean",
                    cfg.clean, frame_range=(lo, hi),
                    on_progress=lambda *a, **kw: _emit(on_progress, *a, **kw),
                    should_cancel=should_cancel,
                )
            except CleanError as exc:
                raise Cancelled(str(exc)) if "取消" in str(exc) else RuntimeError(str(exc)) from exc
            _emit(on_progress, "clean", 1, 1, "旁人已清除")
            check_cancel()

            # 清除旁人模式下，cleaned.mkv 已经是「保留目标 + 抹掉旁人并补回背景」的成品，
            # 整段都应保持清晰——不再做虚化/压暗/去色，也不受「存在门」影响
            # （存在门在目标离场时会切回原片，反而会让被清除的人重新冒出来）。
            # 这里只把无损中间产物重新编码，并合入原片音轨。
            _emit(on_progress, "render", 0, 1, "合成并编码（清除旁人：背景保持清晰）")
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            reencode_pair(
                cleaned, video_path, out_path, info,
                encoder=cfg.encoder, quality=cfg.quality, audio=cfg.audio,
            )
            _emit(on_progress, "render", 1, 1, "完成")
        else:
            # ------------------------------------------------------ 补齐 + 写掩码（虚化模式专用）
            _emit(on_progress, "mask", 0, 2, "补齐掩码并写入序列")
            masks.finalize(interpolate=cfg.frame_step > 1)
            if cfg.frame_start or cfg.frame_end:
                # 用户只处理了一段：范围之外保持空白，让那些帧原样输出
                outside = np.ones(n, dtype=bool)
                outside[lo:hi] = False
                masks.clear(np.flatnonzero(outside))
            mask_path = work / "mask.mkv"
            write_mask_video(masks.iter_frames(), mask_path, store.width, store.height, info.fps_expr, total=n)
            check_cancel()

            _emit(on_progress, "mask", 1, 2, "统计并写入存在门序列")
            min_pixels = max(64, int(0.0006 * store.width * store.height))
            scan = masks.scan(min_pixels)
            levels = gate_levels(scan["present"], cfg.gate_ramp_up, cfg.gate_ramp_down)
            gate_path = work / "gate.mkv"
            write_mask_video(
                (np.full((store.height, store.width), int(v), dtype=np.uint8) for v in levels),
                gate_path, store.width, store.height, info.fps_expr, total=n,
            )
            check_cancel()

            # ------------------------------------------------------ 合成（虚化 + 提亮）
            _emit(on_progress, "render", 0, 1,
                  "合成并编码（景深虚化 + 提亮）" if depth_path else "合成并编码（虚化 + 提亮）")
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            render(
                video_path, mask_path, gate_path, out_path, info,
                mask_size=(store.width, store.height), params=cfg.effects,
                encoder=cfg.encoder, quality=cfg.quality, audio=cfg.audio,
                depth_video=depth_path,
            )
            _emit(on_progress, "render", 1, 1, "完成")

        elapsed = time.perf_counter() - t_start
        if cfg.remove_others:
            # 清除模式不依赖虚化掩码统计
            stats: dict = {}
            lost: list = []
        else:
            stats = summarize_masks(scan["areas"], scan["present"])
            lost = lost_segments(scan["present"])
        history = guard.history[-cfg.timeline_limit :]
        masks.close()
        return RunResult(
            output_path=str(out_path),
            info=info,
            n_frames=n,
            mask_size=(store.width, store.height),
            elapsed=elapsed,
            frames_per_second=(n / elapsed) if elapsed > 0 else 0.0,
            guard_summary=guard.summary(),
            mask_stats=stats,
            lost_segments=lost,
            timeline=history,
            clean_stats=clean_stats,
        )
    finally:
        if own_work:
            shutil.rmtree(work, ignore_errors=True)
