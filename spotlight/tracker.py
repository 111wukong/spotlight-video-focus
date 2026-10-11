"""目标守卫：确保整段视频里始终是「同一个人」，宁可不出效果也不换人。

SAM2 的记忆库很强，但会以几种方式失手：

* **掩码塌缩**：在拥挤或遮挡场景里，掩码可能退化成只覆盖头部的一小块，
  或者干脆变空。
* **交叉穿越**：掩码整体跳到另一个人身上，而面积、位置看起来都还「合理」。
* **长时间出画再回来**：记忆衰减，回来时锁定到旁边相似的人。

这里的处理原则是把两类信号分开用：

* **几何信号（掩码本身好不好）** —— 决定「什么时候该重新锚定」。
* **外观信号（还是不是同一个人）** —— 决定「允许锚定到谁」。

之所以要分开，是因为单独看面积判断不出「人物被挡住了所以只剩一半」
和「模型塌缩了」这两种情况；而单独看外观又发现不了掩码塌缩（人还是那个人，
只是没框全）。合起来用就清楚了：掩码不好就重新锚定，但新锚点必须通过外观比对，
否则宁可保持「丢失」。

模板只在「像自己」的帧上做 EMA 更新，不会被背景或别人污染。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .appearance import (
    AppearanceEngine,
    PersonBox,
    bhattacharyya,
    best_match,
    box_iou,
    cosine,
    mask_centroid,
    mask_to_box,
)
from .segmenter import logits_to_soft


class GuardState(str, Enum):
    OK = "ok"
    SUSPECT = "suspect"
    LOST = "lost"


@dataclass
class GuardConfig:
    """阈值集合。默认值在 1080p 街头/室内人群素材上校准过。"""

    # ---- 几何：掩码可用性 ----
    #: 面积低于参考中位数的这个比例 → 认为掩码塌缩
    area_floor_ratio: float = 0.32
    #: 面积高于参考中位数的这个倍数 → 认为掩码炸开
    area_ceil_ratio: float = 4.5
    #: 参考面积中位数的滑动窗口
    area_window: int = 150
    #: 质心跳动超过「目标尺寸 × 该系数」→ 认为掩码瞬移
    jump_factor: float = 1.0
    #: SAM2 自报的掩码置信度低于此值 → 认为掩码不可靠
    mask_iou_floor: float = 0.25

    # ---- 耐心值：连续多少帧「掩码不可用」才真正动手 ----
    geometry_patience: int = 9
    # ---- 身份：连续多少次窗口判定「不像」才认定丢失 ----
    identity_patience: int = 2

    # ---- 身份看门狗 ----
    identity_every: int = 6
    identity_window: int = 5
    #: 窗口内平均外观相似度低于此值 → 判定「不像本人」
    identity_floor: float = 0.45

    # ---- 重锚定 ----
    #: 允许作为新锚点的最低综合外观相似度
    reanchor_score: float = 0.60
    #: 两次重锚定尝试之间至少间隔多少帧
    reanchor_interval: int = 3
    #: 人体检测的最低置信度（候选池宁可多不可少）
    detect_min_confidence: float = 0.12
    #: 丢失超过这么多帧后，先清空被污染的记忆再重锚定
    purge_after: int = 20

    # ---- 模板 ----
    template_momentum: float = 0.88
    #: 只有相似度高于此值才允许更新模板
    template_trust: float = 0.50
    #: 外观特征与颜色直方图在综合得分中的权重
    fp_weight: float = 0.65


@dataclass
class GuardResult:
    soft_mask: np.ndarray
    state: GuardState
    similarity: float = -1.0
    mask_iou: float = -1.0
    area_ratio: float = 0.0
    reanchor_box: PersonBox | None = None
    note: str = ""
    purge_memory: bool = False

    @property
    def healthy(self) -> bool:
        return self.state is GuardState.OK


class TargetGuard:
    """逐帧观察掩码，输出「可以直接拿去合成」的掩码 + 是否需要重锚定。"""

    def __init__(self, engine: AppearanceEngine, config: GuardConfig | None = None) -> None:
        self.engine = engine
        self.cfg = config or GuardConfig()
        self._areas: deque[int] = deque(maxlen=self.cfg.area_window)
        self._sims: deque[float] = deque(maxlen=self.cfg.identity_window)
        self.fp_template: np.ndarray | None = None
        self.hist_template: np.ndarray | None = None
        self.last_box: PersonBox | None = None
        self.last_centroid: tuple[float, float] | None = None
        self.last_good_mask: np.ndarray | None = None
        self.geo_run = 0
        self.identity_run = 0
        self.lost_run = 0
        self.total_suspect = 0
        self.total_lost = 0
        self.reanchors = 0
        self._last_reanchor_at = -10**9
        self._frame_seen = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ 初始化

    def seed(self, rgb: np.ndarray, logits: np.ndarray) -> None:
        """用用户选定那一帧的掩码建立初始模板。"""
        soft = logits_to_soft(logits)
        binary = logits > 0.0
        area = int(binary.sum())
        self._areas.clear()
        self._sims.clear()
        if area > 0:
            self._areas.append(area)
        box = mask_to_box(soft)
        if box is not None:
            self.last_box = box
            self.last_centroid = mask_centroid(soft)
            self.last_good_mask = soft.copy()
            fp = self.engine.embed_masked(rgb, soft)
            if fp is not None:
                self.fp_template = fp
            self.hist_template = self.engine.color_hist(rgb, soft)
        self.geo_run = self.identity_run = self.lost_run = 0
        self.total_suspect = self.total_lost = 0
        self._frame_seen = 0

    # ------------------------------------------------------------------ 主循环

    def observe(
        self,
        frame_idx: int,
        rgb: np.ndarray,
        logits: np.ndarray,
        mask_iou: float = -1.0,
    ) -> GuardResult:
        cfg = self.cfg
        self._frame_seen += 1
        soft = logits_to_soft(logits)
        area = int((logits > 0.0).sum())
        h, w = logits.shape[:2]
        min_area = max(48, int(0.00025 * h * w))
        box = mask_to_box(soft)
        centroid = mask_centroid(soft)
        ref_area = float(np.median(self._areas)) if len(self._areas) >= 5 else None

        # ---------------- 几何信号：掩码本身可用吗
        geo_bad, geo_note = False, ""
        if area < min_area or box is None:
            geo_bad, geo_note = True, "掩码为空"
        elif ref_area and area < ref_area * cfg.area_floor_ratio:
            geo_bad, geo_note = True, f"掩码塌缩到 {area / ref_area:.0%}"
        elif ref_area and area > ref_area * cfg.area_ceil_ratio:
            geo_bad, geo_note = True, f"掩码炸开到 {area / ref_area:.1f}x"
        elif 0.0 <= mask_iou < cfg.mask_iou_floor:
            geo_bad, geo_note = True, f"掩码置信度低 ({mask_iou:.2f})"
        elif self.last_centroid is not None and centroid is not None and self.last_box is not None:
            jump = float(np.hypot(centroid[0] - self.last_centroid[0], centroid[1] - self.last_centroid[1]))
            scale = max(self.last_box.width, self.last_box.height, 8.0)
            if jump > cfg.jump_factor * scale:
                geo_bad, geo_note = True, f"质心瞬移 {jump:.0f}px"

        # ---------------- 外观信号：还是不是同一个人
        sim = -1.0
        if self.fp_template is not None:
            need = (
                geo_bad
                or self.geo_run > 0
                or self.identity_run > 0
                or self._frame_seen % cfg.identity_every == 0
            )
            if need:
                crop = soft if box is not None else self.last_good_mask
                fp = self.engine.embed_masked(rgb, crop) if crop is not None else None
                if fp is not None:
                    sim = cosine(self.fp_template, fp)
                    self._sims.append(sim)
        identity_bad = len(self._sims) >= cfg.identity_window and float(np.mean(self._sims)) < cfg.identity_floor

        self.geo_run = self.geo_run + 1 if geo_bad else 0
        self.identity_run = self.identity_run + 1 if identity_bad else 0

        # ---------------- 判定
        if not geo_bad and not identity_bad:
            self.lost_run = 0
            self._commit(rgb, soft, area, box, centroid, sim)
            self.history.append({"frame": frame_idx, "state": "ok", "sim": round(sim, 3), "area": area})
            return GuardResult(soft, GuardState.OK, sim, mask_iou, 1.0)

        # 掩码不好但人还是本人 → 先原样输出，别把画面搞花
        if self.identity_run < cfg.identity_patience and self.geo_run < cfg.geometry_patience:
            self.total_suspect += 1
            self.history.append({"frame": frame_idx, "state": "suspect", "sim": round(sim, 3), "note": geo_note})
            return GuardResult(soft, GuardState.SUSPECT, sim, mask_iou, 0.0, note=geo_note)

        # 到了该重新锚定的时候
        self.lost_run += 1
        self.total_lost += 1
        purge = self.lost_run > cfg.purge_after and self.lost_run % 5 == 0
        note = geo_note or (f"外观不像本人 ({np.mean(self._sims):.2f})" if identity_bad else "目标丢失")

        if frame_idx - self._last_reanchor_at >= cfg.reanchor_interval:
            self._last_reanchor_at = frame_idx
            found, score, why = self._try_reanchor(rgb)
            if found is not None:
                self.geo_run = self.identity_run = self.lost_run = 0
                self._sims.clear()
                self.reanchors += 1
                self.history.append({"frame": frame_idx, "state": "reanchor", "score": round(score, 3), "note": why})
                return GuardResult(
                    np.zeros_like(soft), GuardState.LOST, score, mask_iou, 0.0,
                    reanchor_box=found, note=why, purge_memory=purge,
                )
            note = why

        # 掩码塌缩但人没认错：保留原掩码总比整帧虚化好
        if self.identity_run < cfg.identity_patience:
            self.total_suspect += 1
            self.history.append({"frame": frame_idx, "state": "suspect", "note": note})
            return GuardResult(soft, GuardState.SUSPECT, sim, mask_iou, 0.0, note=note, purge_memory=purge)

        self.history.append({"frame": frame_idx, "state": "lost", "note": note})
        return GuardResult(np.zeros_like(soft), GuardState.LOST, sim, mask_iou, 0.0, note=note, purge_memory=purge)

    # ------------------------------------------------------------------ 手动接管

    def accept_manual(self, frame_idx: int, rgb: np.ndarray, logits: np.ndarray, momentum: float = 0.3) -> None:
        """采信用户在某一帧手动指定的目标。

        跟踪跑偏时用户会自己补一个修正点，这一帧的掩码是「标准答案」：
        计数全部清零（不要把之前的异常延续下来），并把外观模板按较大的权重
        向这一帧靠拢——用户显式指认的置信度比模型自己跟踪出来的高得多。
        """
        soft = logits_to_soft(logits)
        area = int((logits > 0.0).sum())
        if area <= 0:
            return
        box = mask_to_box(soft)
        centroid = mask_centroid(soft)
        self.geo_run = self.identity_run = self.lost_run = 0
        self._sims.clear()
        if box is not None:
            self.last_box = box
            self.last_centroid = centroid
            self.last_good_mask = soft.copy()
        self._areas.append(area)
        fp = self.engine.embed_masked(rgb, soft)
        if fp is not None:
            if self.fp_template is None:
                self.fp_template = fp
            else:
                blended = (1.0 - momentum) * self.fp_template + momentum * fp
                norm = float(np.linalg.norm(blended))
                if norm > 1e-6:
                    self.fp_template = blended / norm
        hist = self.engine.color_hist(rgb, soft)
        if hist is not None and hist.size:
            if self.hist_template is None:
                self.hist_template = hist
            else:
                self.hist_template = (1.0 - momentum) * self.hist_template + momentum * hist
        self.history.append({"frame": frame_idx, "state": "manual", "area": area})

    # ------------------------------------------------------------------ 内部

    def _commit(self, rgb, soft, area, box, centroid, sim) -> None:
        """记录一次「健康」的观察：更新参考面积、位置，以及外观模板。"""
        cfg = self.cfg
        self._areas.append(area)
        self.last_box = box
        self.last_centroid = centroid
        self.last_good_mask = soft.copy()
        if sim >= cfg.template_trust or self.fp_template is None:
            fp = self.engine.embed_masked(rgb, soft)
            hist = self.engine.color_hist(rgb, soft)
            m = cfg.template_momentum
            if fp is not None:
                self.fp_template = fp if self.fp_template is None else (m * self.fp_template + (1 - m) * fp)
                norm = float(np.linalg.norm(self.fp_template))
                if norm > 1e-6:
                    self.fp_template = self.fp_template / norm
            if hist is not None and hist.size:
                self.hist_template = hist if self.hist_template is None else (
                    m * self.hist_template + (1 - m) * hist
                )

    def _try_reanchor(self, rgb: np.ndarray) -> tuple[PersonBox | None, float, str]:
        """在画面里找最像目标的人体框。"""
        cfg = self.cfg
        try:
            people = self.engine.detect_people(rgb, min_confidence=cfg.detect_min_confidence)
        except Exception as exc:  # pragma: no cover - Vision 异常不应中断整个流程
            return None, -1.0, f"人体检测失败：{exc}"
        if not people:
            return None, -1.0, "画面里没有检测到人"

        cands: list[tuple[np.ndarray | None, np.ndarray]] = []
        for p in people:
            cands.append((self.engine.embed(rgb, p), self.engine.color_hist(rgb, _box_mask(p, rgb.shape[:2]))))
        idx, score = best_match(self.fp_template, self.hist_template, cands, fp_weight=cfg.fp_weight)
        if idx < 0:
            return None, score, "候选无法比对"

        # 位置合理性：刚丢失时目标不该瞬间跑到画面另一头
        if self.last_centroid is not None and self.lost_run <= 3:
            p = people[idx]
            c = ((p.x1 + p.x2) / 2, (p.y1 + p.y2) / 2)
            h, w = rgb.shape[:2]
            if np.hypot(c[0] - self.last_centroid[0], c[1] - self.last_centroid[1]) > 0.75 * np.hypot(w, h):
                return None, score, "最像的候选位置过远"
        if score < cfg.reanchor_score:
            return None, score, f"最佳候选仅 {score:.2f}，低于阈值"
        return people[idx], score, f"重锚定成功 (相似度 {score:.2f})"

    def summary(self) -> dict:
        return {
            "reanchors": self.reanchors,
            "suspect_frames": self.total_suspect,
            "lost_frames": self.total_lost,
            "frames_seen": self._frame_seen,
        }


def _box_mask(box: PersonBox, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = int(max(0, box.x1)), int(max(0, box.y1))
    x2, y2 = int(min(w, box.x2)), int(min(h, box.y2))
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 255
    return m
