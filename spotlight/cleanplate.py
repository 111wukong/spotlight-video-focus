"""清除旁人：把画面里除目标外的所有人抹掉，并用重建出来的背景补上。

整体思路
--------
1. **移除掩码** = 全体人员掩码（Vision 人像分割）− 目标掩码（SAM2），再膨胀一点。
2. **背景重建** = 运动补偿的分段时域累积。每一段在段中点坐标系里把所有帧对齐叠加，
   只取「完全没有人的像素」，得到一张干净的背景板；空洞交给 LaMa 修复。
3. **合成** = 只在移除区域内用背景板替换，区域外保留原像素（目标因此天然保留）。

为什么是「分段板」而不是「逐帧重建」
------------------------------------
* 逐帧重建每帧都要重新累积一个时间窗，成本高；更糟的是每帧的板略有不同，
  合成出来会**闪**。
* 分段板每段只累积一次、段内所有帧共用同一张板，**段内完全时间一致**，
  不会有任何闪烁；段长控制在 64 帧左右，累积漂移可以忽略。
* 段边界用相邻两张板按时间权重交叉淡化，避免切换时跳变。

空洞为什么会存在
----------------
如果某块背景在整个段里始终被人挡着，就永远采不到样本——这时才需要 LaMa 去「编」。
人群越密、段越短，空洞越多；实测密集人群 64 帧段约 12~20%。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .appearance import PersonSegmenter
from .video import FrameStore, MaskStore, write_frame_video

ProgressFn = Callable[..., None]


class CleanError(RuntimeError):
    pass


@dataclass
class CleanConfig:
    """清除旁人的参数。"""

    #: 每多少帧换一张背景板（控制时间一致性：换得越勤越不易「内容过期」，
    #: 但相邻板之间的差异也越需要靠交叉淡化来抹平）
    segment: int = 32
    #: 每张板取中心帧前后各多少帧的样本。**这是视差与空洞的关键权衡**：
    #: 单应矩阵只能对齐一个平面，帧距越远视差越大、错位越明显。
    #: 实测（真实人群素材 960x540）：
    #:     窗口 ±12 → 一致性误差 4.00，空洞 12.2%
    #:     窗口 ±16 → 一致性误差 3.57，空洞 11.6%   ← 最优点
    #:     窗口 ±24 → 一致性误差 4.39，空洞 11.2%
    #:     窗口 ±64 → 一致性误差 7.29，空洞  8.6%   ← 误差翻倍
    #: 也就是说「窗口开大」换来的那点空洞收益，远不及它带来的重影代价。
    window: int = 16
    #: 移除掩码膨胀像素数（避免人物边缘残留一圈）
    dilate: int = 7
    #: 合成时边缘羽化
    feather: float = 2.5
    #: 一个像素少于这么多个样本就算空洞，需要 LaMa 补
    min_samples: int = 3
    #: 是否用 LaMa 修复空洞（关掉的话空洞会留黑）
    inpaint: bool = True
    #: 补洞用的模型：``lama``（默认，快）或 ``diffusion``（SD inpainting，质量更高但慢）。
    #: 选 ``diffusion`` 需要先用 ``spotlight download --preset diffusion`` 拉权重。
    inpaint_model: str = "lama"
    #: 人员分割质量档：accurate / balanced / fast
    person_quality: str = "accurate"
    #: 判定「这个像素没人」时，把人员掩码再胀这么多像素。
    #: 人像分割的边缘是软的、还会漏掉远处的小人，不胀一点的话，
    #: 那些「看起来像背景、其实还是人」的像素会被当成背景样本累积进来，
    #: 最后在成片里表现为一层半透明的「人影残像」。
    person_pad: int = 9
    #: 时域标准差超过这个值就判为不可信（当作空洞交给 LaMa）。
    #: **默认关闭**：实测在移动镜头下，视差和插值本身就会让所有像素剧烈波动，
    #: 这个判据会把空洞从 5% 推到 24%，而背景一致性并没有改善（7.55 → 7.68）。
    #: 残像问题改由下面的 ``robust_tol``（两步截尾均值）解决。
    unstable_std: float = 0.0
    #: 稳健估计容差（灰阶）。>0 时启用两步截尾均值：
    #: 先用加权均值得到初值，再只采信「偏离初值不超过这个容差」的样本。
    #: 这一步专门对付**漏检**——人像分割难免漏掉远处的小人，那些像素会被当成
    #: 背景样本参与平均，累积出来就是半透明的「人影残像」。截尾能把它们踢掉。
    #: 设为 0 则退化成普通加权均值。
    robust_tol: float = 40.0
    #: 单应估计时用的特征点数
    features: int = 2000
    #: RANSAC 重投影阈值
    ransac_px: float = 3.0
    #: 段间是否交叉淡化
    crossfade: bool = True
    #: 补洞时可以借用的邻近段板范围（单位：段长）。越大覆盖越全，但借得越远、
    #: 单应链的漂移越大，搬过来的内容越不可信。实测 1~8 之间误差几乎不变，
    #: 但空洞能从 36% 降到 27%，所以默认取大一点。
    donor_span: int = 8
    #: 保护区域的总量预算 = 目标掩码面积 × 这个系数。整套兜底逻辑的保险丝：
    #: 目标掩码和人像分割都可能出问题，没有预算的话任何一条兜底规则都可能
    #: 把半个画面保下来，等于什么都没清。2.0 大致够「身体 + 头」，又不足以吞掉旁边的人。
    keep_ratio: float = 2.0
    #: 用人体检测框来兜住目标的完整身体（含头）。关掉的话只靠掩码和外接框。
    use_person_box: bool = True
    #: 判定「这个人体框就是目标」所需的与目标掩码的重叠比例（相对目标面积）
    person_box_min_hit: float = 0.35
    #: 目标外接框向外扩的比例（相对目标自身宽高）。
    #: 用来兜住 SAM2 没切到、且在人员掩码里和目标**断开**的部位（典型是头）。
    #: 这是刻意偏保守的：宁可少抹掉一个人，也不能把目标自己弄坏。
    keep_margin: float = 0.2


# --------------------------------------------------------------------- LaMa


class LamaInpainter:
    """LaMa 图像修复（ONNX，走 CoreML / ANE）。

    LaMa 是「大掩码修复」的经典模型，对整块缺失区域的补全质量远好于
    OpenCV 自带的 ``inpaint``（后者会把大块区域糊成一团）。
    这里只在**背景板**上跑——每段一次，所以成本很低，而且段内天然时间一致。
    """

    def __init__(self, model_path: str | Path, threads: int = 4) -> None:
        self.model_path = Path(model_path)
        self.available = False
        self.reason = ""
        self._session = None
        self._input_names: list[str] = []
        self._size = 512
        try:
            import onnxruntime as ort

            providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                         if p in ort.get_available_providers()]
            options = ort.SessionOptions()
            options.intra_op_num_threads = int(threads)
            options.log_severity_level = 3
            self._session = ort.InferenceSession(
                str(self.model_path), sess_options=options, providers=providers
            )
            self._input_names = [i.name for i in self._session.get_inputs()]
            self.available = True
        except Exception as exc:  # noqa: BLE001 - 没装 onnxruntime 或没下模型都算不可用
            self.reason = f"{type(exc).__name__}: {exc}"

    def fill(self, rgb: np.ndarray, hole: np.ndarray) -> np.ndarray:
        """把 ``hole``（非零=要补）区域补上，返回补好的图。失败时原样返回。"""
        if not self.available or not hole.any():
            return rgb
        h, w = rgb.shape[:2]
        size = self._size
        img = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(hole, (size, size), interpolation=cv2.INTER_NEAREST)
        msk = (msk > 127).astype(np.float32)

        image_in = img.astype(np.float32) / 255.0
        image_in = np.transpose(image_in, (2, 0, 1))[None]
        mask_in = msk[None, None]
        feed = {}
        for name in self._input_names:
            lowered = name.lower()
            feed[name] = mask_in if ("mask" in lowered) else image_in
        if len(feed) == 1:
            feed[self._input_names[0]] = image_in

        try:
            outputs = self._session.run(None, feed)
        except Exception:  # noqa: BLE001 - 单帧修复失败不应该中断整段流程
            return rgb

        out = np.asarray(outputs[0])
        out = np.squeeze(out)
        if out.ndim == 3 and out.shape[0] == 3:
            out = np.transpose(out, (1, 2, 0))
        out = np.clip(out * 255.0 if out.max() <= 1.5 else out, 0, 255).astype(np.uint8)
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)

        # 只取空洞区域的结果，其余保留原像素（LaMa 会轻微改变全图）
        alpha = (cv2.GaussianBlur(hole, (0, 0), 2.0).astype(np.float32) / 255.0)[..., None]
        return (rgb.astype(np.float32) * (1 - alpha) + out.astype(np.float32) * alpha).astype(np.uint8)


def resolve_lama(models_dir: str | Path) -> Path | None:
    for candidate in (
        Path(models_dir) / "lama" / "lama_fp32.onnx",
        Path(models_dir) / "lama" / "lama.onnx",
        Path(models_dir) / "lama_fp32.onnx",
    ):
        if candidate.is_file():
            return candidate
    return None


# --------------------------------------------------------------------- 运动估计


def _frame_homographies(
    store: FrameStore,
    persons: MaskStore,
    cfg: CleanConfig,
    on_progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> list[np.ndarray]:
    """估计相邻帧之间的单应矩阵 ``H[i]``（把第 i-1 帧对齐到第 i 帧）。

    只在**完全没有人的区域**上提特征，否则路人的运动会把估计带偏。
    """
    orb = cv2.ORB_create(cfg.features)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    n = store.n_frames
    homographies: list[np.ndarray] = [np.eye(3, dtype=np.float64)]
    prev_gray = cv2.cvtColor(store.frame(0), cv2.COLOR_RGB2GRAY)
    prev_keep = (persons.get(0) < 100).astype(np.uint8) * 255
    failures = 0

    for i in range(1, n):
        if should_cancel is not None and should_cancel():
            raise CleanError("已取消")
        cur_gray = cv2.cvtColor(store.frame(i), cv2.COLOR_RGB2GRAY)
        cur_keep = (persons.get(i) < 100).astype(np.uint8) * 255
        homography = None
        try:
            k1, d1 = orb.detectAndCompute(prev_gray, prev_keep)
            k2, d2 = orb.detectAndCompute(cur_gray, cur_keep)
            if d1 is not None and d2 is not None and len(k1) >= 12 and len(k2) >= 12:
                matches = matcher.match(d1, d2)
                if len(matches) >= 12:
                    p1 = np.float32([k1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    p2 = np.float32([k2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    homography, _ = cv2.findHomography(p1, p2, cv2.RANSAC, cfg.ransac_px)
        except cv2.error:
            homography = None
        if homography is None or not np.isfinite(homography).all():
            homography = np.eye(3, dtype=np.float64)
            failures += 1
        homographies.append(np.asarray(homography, dtype=np.float64))
        prev_gray, prev_keep = cur_gray, cur_keep

    if on_progress is not None:
        on_progress("motion", n, n, f"运动估计完成（{failures} 帧回退为单位矩阵）",
                    motion_failures=failures)
    return homographies


def _accumulate_homography(homographies: list[np.ndarray], src: int, dst: int) -> np.ndarray:
    """把第 ``src`` 帧的坐标系变换到第 ``dst`` 帧。"""
    if src == dst:
        return np.eye(3, dtype=np.float64)
    result = np.eye(3, dtype=np.float64)
    if src < dst:
        for j in range(src + 1, dst + 1):
            result = homographies[j] @ result
    else:
        for j in range(dst + 1, src + 1):
            result = np.linalg.inv(homographies[j]) @ result
    return result


# --------------------------------------------------------------------- 主流程


class PeopleRemover:
    """把除目标外的所有人从画面里抹掉。"""

    def __init__(
        self,
        cfg: CleanConfig | None = None,
        inpainter: LamaInpainter | None = None,
        engine: "AppearanceEngine | None" = None,
    ) -> None:
        self.cfg = cfg or CleanConfig()
        self.inpainter = inpainter
        self.engine = engine
        self.stats: dict = {}

    # ---------------------------------------------------------------- 掩码

    def build_person_masks(
        self,
        store: FrameStore,
        work: Path,
        on_progress: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> MaskStore:
        """逐帧分割出画面里所有的人。"""
        segmenter = PersonSegmenter(self.cfg.person_quality)
        if not segmenter.available:
            raise CleanError("系统的人像分割不可用（需要 macOS + Vision 框架）")
        masks = MaskStore(work / "persons.u8", store.n_frames, store.width, store.height)
        for i in range(store.n_frames):
            if should_cancel is not None and should_cancel():
                raise CleanError("已取消")
            person = segmenter.mask(store.frame(i), (store.width, store.height))
            if person is None:
                person = np.zeros((store.height, store.width), dtype=np.uint8)
            masks.set(i, person)
            if on_progress is not None and (i % 5 == 0 or i == store.n_frames - 1):
                on_progress("people", i + 1, store.n_frames, f"分割画面里的人 {i + 1}/{store.n_frames}")
        return masks

    def _target_region(
        self,
        person: np.ndarray,
        target: np.ndarray | None,
        kernel: np.ndarray,
        rgb: np.ndarray | None = None,
    ) -> np.ndarray:
        """算出「属于目标、不能抹掉」的区域。

        为什么要这么绕，而不是直接「目标掩码膨胀一圈」：SAM2 常常只分割出目标的
        一部分——典型情况是**只切了身体、没切到头部**（头被 VR 眼镜、口罩、帽子
        这类物体挡住时尤其明显）。只保护掩码那一块的话，目标的头会被当成「别人」
        抹掉，画面里就会出现「身体还在、头被换成了背景」这种很诡异的结果。

        分三层兜，从精确到宽松：
        1. **人体检测框** —— 用系统的人体矩形检测找到目标所在的那个框，
           把框内的「人员像素」一并保留。检测框天然包含头部，而且一个人一个框，
           不会像「外接框外扩」那样把旁边的人一起圈进来。
        2. **连通域整块保留** —— 目标和它的某个部位在「全体人员」掩码里可能是连着的，
           只要某个连通域和目标有足够重叠，就整块保下来。
        3. **外接框小幅外扩** —— 兜底，防止 1、2 都没命中。

        整体方向是刻意偏保守的：**宁可少抹掉一个人，也不能把目标自己弄坏**——
        少抹一个只是效果打折，把目标弄坏则整个结果作废。
        """
        if target is None or not (target > 127).any():
            return np.zeros_like(person)
        h, w = person.shape[:2]
        binary = (person > 127).astype(np.uint8)
        target_mask = target > 127
        target_area = int(target_mask.sum())
        # 目标掩码本身永远保护——它是对「目标是谁」的唯一定义，不能因为预算被裁掉。
        keep = target_mask.copy()
        # 预算约束的是**额外**兜底的部分：目标掩码 + 兜底 ≤ 目标面积 × keep_ratio。
        # 这一条是整套逻辑的保险丝：目标掩码和人像分割都可能出问题（掩码泄漏、
        # 把旁边的人糊进来），没有预算的话任何一条兜底规则都可能把半个画面保下来，
        # 等于什么都没清。
        budget = max(4096, int(target_area * self.cfg.keep_ratio))
        used = int(target_area)

        def absorb(candidate: np.ndarray) -> None:
            """把候选区域并进保护范围，但不能超出预算。"""
            nonlocal used
            extra = np.asarray(candidate).astype(bool) & ~keep
            count = int(np.count_nonzero(extra))
            if used + count > budget:
                return
            keep[extra] = True
            used += count

        # 1) 与目标重叠的连通域整块保留（最可靠：目标和它的某个部位连在一起）
        count, labels = cv2.connectedComponents(binary, connectivity=8)
        if count > 1:
            for label in np.unique(labels[target_mask]):
                label = int(label)
                if label == 0:
                    continue
                comp = labels == label
                area = int(comp.sum())
                if area > budget:
                    continue
                overlap = int(np.count_nonzero(comp & target_mask))
                # 「这块主要就是目标」或「目标主要就在这块里」都算
                if overlap >= 0.2 * area or overlap >= 0.6 * target_area:
                    absorb(comp)

        # 2) 目标所在的人体检测框 ∩ 人员掩码。
        #    检测框天然包含头部，而且一个人一个框，比「外接框外扩」精确得多。
        if self.engine is not None and rgb is not None and self.cfg.use_person_box:
            try:
                people = self.engine.detect_people(rgb)
            except Exception:  # noqa: BLE001 - 检测失败不应该中断流程
                people = []
            # 在「和目标掩码有足够重叠」的框里挑**最小的那个**：
            # 目标离镜头近时检测器会给出几乎覆盖全画面的大框，
            # 用它就把旁边的人一起保护掉了。最小的合格框通常正好是目标本人。
            best, best_area = None, None
            for box in people:
                x1, y1 = max(0, int(box.x1)), max(0, int(box.y1))
                x2, y2 = min(w, int(box.x2)), min(h, int(box.y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                hit = int(np.count_nonzero(target_mask[y1:y2, x1:x2]))
                if hit < self.cfg.person_box_min_hit * target_area:
                    continue
                area = (x2 - x1) * (y2 - y1)
                if best_area is None or area < best_area:
                    best, best_area = (x1, y1, x2, y2), area
            if best is not None:
                x1, y1, x2, y2 = best
                box = np.zeros_like(person)
                box[y1:y2, x1:x2] = 255
                absorb(person & box)

        # 3) 外接框小幅外扩兜底
        margin = self.cfg.keep_margin
        if margin > 0:
            ys, xs = np.nonzero(target_mask)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            mx = int(round((x2 - x1) * margin))
            my = int(round((y2 - y1) * margin))
            box = np.zeros_like(person)
            box[max(0, y1 - my):min(h, y2 + my + 1), max(0, x1 - mx):min(w, x2 + mx + 1)] = 255
            absorb(person & box)

        if not keep.any():
            return cv2.dilate(target, kernel, iterations=1)
        return cv2.dilate(keep.astype(np.uint8) * 255, kernel, iterations=1)

    def build_remove_masks(
        self,
        store: FrameStore,
        persons: MaskStore,
        targets: MaskStore,
        work: Path,
        on_progress: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> MaskStore:
        """移除掩码 = 所有人 − 目标（膨胀后）。"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.cfg.dilate, self.cfg.dilate))
        remove = MaskStore(work / "remove.u8", store.n_frames, store.width, store.height)
        for i in range(store.n_frames):
            if should_cancel is not None and should_cancel():
                raise CleanError("已取消")
            person = persons.get(i)
            target = targets.get(i) if i < targets.n else None
            # 目标自己不能被抹掉。这里按连通域保护，而不是只保护掩码那一块——
            # 否则目标身上没被 SAM2 切到的部位（典型是头）会被当成别人抹掉。
            keep = self._target_region(person, target, kernel, rgb=store.frame(i))
            mask = cv2.bitwise_and(person, cv2.bitwise_not(keep))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.dilate(mask, kernel, iterations=1)
            remove.set(i, mask)
            if on_progress is not None and (i % 10 == 0 or i == store.n_frames - 1):
                on_progress("remove", i + 1, store.n_frames, f"计算要抹掉的区域 {i + 1}/{store.n_frames}")
        return remove

    # ---------------------------------------------------------------- 背景板

    def _build_plate(
        self,
        store: FrameStore,
        persons: MaskStore,
        homographies: list[np.ndarray],
        lo: int,
        hi: int,
        reference: int | None = None,
        sigma: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把 ``[lo, hi)`` 累积成一张背景板（默认以段中点为参考系）。

        ``sigma`` 控制时间权重：离参考帧越近的帧权重越高。这一项很重要——
        单应矩阵只能对齐一个平面，走廊、街道这类有纵深的场景必然存在视差，
        帧隔得越远视差越大、错位越明显，所以时间近的样本更可信。

        先做一遍加权均值，可选再做一遍**截尾**（``robust_tol > 0``）：
        只采信偏离初值不超过容差的样本。实测这一步能把背景区的一致性误差
        从 6.0 降到 4.9 灰阶（双平面合成场景），代价是极少量空洞。
        """
        center = (lo + hi) // 2 if reference is None else int(reference)
        h, w = store.height, store.width
        if sigma is None:
            sigma = max(1.0, (hi - lo) / 4.0)

        mean, count = self._accumulate(
            store, persons, homographies, lo, hi, center, sigma
        )

        # 第二步（截尾）：只采信落在初值附近的样本。
        # 目的是把人像分割**漏检**的像素从背景样本里踢出去——它们会被第一步的均值
        # 带偏，累积出来就是半透明的「人影残像」。这一步要再扫一遍帧，所以只在
        # robust_tol > 0 时做。
        if self.cfg.robust_tol > 0:
            mean2, count2 = self._accumulate(
                store, persons, homographies, lo, hi, center, sigma,
                estimate=mean, tol=self.cfg.robust_tol,
            )
            stable = count2 >= max(1, self.cfg.min_samples)
            mean[stable] = mean2[stable]
            count[stable] = count2[stable]
            holes_mask = count2 < self.cfg.min_samples
        else:
            holes_mask = count < self.cfg.min_samples

        plate = np.zeros((h, w, 3), dtype=np.uint8)
        seen = count > 0
        plate[seen] = np.clip(mean[seen], 0, 255).astype(np.uint8)

        holes = holes_mask.astype(np.uint8) * 255
        holes = cv2.morphologyEx(holes, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        return plate, holes

    def _accumulate(
        self,
        store: FrameStore,
        persons: MaskStore,
        homographies: list[np.ndarray],
        lo: int,
        hi: int,
        center: int,
        sigma: float,
        estimate: np.ndarray | None = None,
        tol: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """把 ``[lo, hi)`` 的帧对齐到 ``center`` 帧坐标后做时间加权累积。

        有效样本要同时满足三条：**没被人挡着**、**来自画面内部**、
        （启用截尾时）**与 ``estimate`` 的偏离不超过 ``tol``**。

        第二条很容易漏——``warpPerspective`` 会把画外补成黑色，不排除的话
        这些黑边会被当成有效样本参与平均，整张板就被拉黑了。
        """
        h, w = store.height, store.width
        acc = np.zeros((h, w, 3), dtype=np.float64)
        weight_sum = np.zeros((h, w), dtype=np.float64)
        count = np.zeros((h, w), dtype=np.int32)
        canvas = np.full((h, w), 255, dtype=np.uint8)

        pad = self.cfg.person_pad
        kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
                  if pad > 0 else None)
        trimmed = estimate is not None and tol > 0

        for i in range(lo, hi):
            H = _accumulate_homography(homographies, i, center)
            warped = cv2.warpPerspective(store.frame(i), H, (w, h), flags=cv2.INTER_LINEAR)
            person = cv2.warpPerspective(persons.get(i), H, (w, h), flags=cv2.INTER_NEAREST)
            if kernel is not None:
                person = cv2.dilate(person, kernel)
            inside = cv2.warpPerspective(canvas, H, (w, h), flags=cv2.INTER_NEAREST) == 255
            valid = inside & (person < 60)
            pixels = warped.astype(np.float64)
            if trimmed:
                valid &= np.abs(pixels - estimate).max(axis=2) <= tol
            weight = float(np.exp(-abs(i - center) / sigma))
            v = valid.astype(np.float64) * weight
            acc += pixels * v[..., None]
            weight_sum += v
            count += valid

        mean = np.zeros((h, w, 3), dtype=np.float64)
        seen = weight_sum > 0
        mean[seen] = acc[seen] / weight_sum[seen][:, None]
        return mean, count

    def _fill_from_donor(
        self,
        plate: np.ndarray,
        holes: np.ndarray,
        donor_plate: np.ndarray,
        donor_holes: np.ndarray,
        donor_center: int,
        center: int,
        homographies: list[np.ndarray],
        size: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """把 ``donor_plate`` 上「有数据」的像素搬进 ``plate`` 的空洞里。

        为什么要这么做：某个位置在这几十帧里一直被人挡着，但在别的时段
        （镜头挪开了）是露出来过的。搬过来的是**真实画面**，比让 LaMa 凭空编可信。

        **关键**：只能从 donor「有数据」的地方取。donor 自己也有空洞（黑色），
        不排除的话会把黑色一起搬过来，画面会莫名其妙变暗——这是踩过的坑。
        """
        w, h = size
        H = _accumulate_homography(homographies, donor_center, center)
        warped = cv2.warpPerspective(donor_plate, H, (w, h), flags=cv2.INTER_LINEAR)
        warped_holes = cv2.warpPerspective(donor_holes, H, (w, h), flags=cv2.INTER_NEAREST)
        canvas = np.full((h, w), 255, dtype=np.uint8)
        inside = cv2.warpPerspective(canvas, H, (w, h), flags=cv2.INTER_NEAREST) == 255
        fillable = (holes > 127) & inside & (warped_holes < 127)
        if not fillable.any():
            return plate, holes
        out = plate.copy()
        out[fillable] = warped[fillable]
        remaining = holes.copy()
        remaining[fillable] = 0
        return out, remaining

    def build_plates(
        self,
        store: FrameStore,
        persons: MaskStore,
        homographies: list[np.ndarray],
        frame_range: tuple[int, int] | None = None,
        on_progress: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> tuple[list[tuple[int, np.ndarray]], float]:
        """按段建背景板，并顺手补上空洞。

        返回 ``[(段中心帧号, 背景板), ...]`` 和平均空洞占比。
        """
        n = store.n_frames
        seg = max(8, int(self.cfg.segment))
        win = max(4, int(self.cfg.window))
        bounds: list[tuple[int, int]] = []
        start = max(0, int(frame_range[0])) if frame_range else 0
        stop = min(n, int(frame_range[1])) if frame_range else n
        lo = start
        while lo < stop:
            hi = min(stop, lo + seg)
            bounds.append((lo, hi))
            lo = hi

        # 第一步：把所有段板都建出来。
        # 每张板只用自己的时间窗（默认 ±16 帧），视差最小、最准；
        # 取样窗口刻意不受 frame_range 限制——多拿到的都是有效观测。
        raw: list[list] = []
        for k, (lo, hi) in enumerate(bounds):
            if should_cancel is not None and should_cancel():
                raise CleanError("已取消")
            center = (lo + hi) // 2
            plate, holes = self._build_plate(
                store, persons, homographies,
                max(0, center - win), min(n, center + win + 1),
                reference=center, sigma=max(2.0, win / 2.0),
            )
            raw.append([center, plate, holes])
            if on_progress is not None:
                on_progress(
                    "plate", k + 1, len(bounds),
                    f"重建背景板 {k + 1}/{len(bounds)}（空洞 {float((holes > 127).mean()):.0%}）",
                    hole_ratio=round(float((holes > 127).mean()), 4),
                )

        # 第二步：段板之间互相补洞，按时间距离从近到远。
        #
        # 这里**不用**「一张全片板」去补。原因是踩过的坑：全片板要以某一帧为参考系
        # 累积几百帧，单应链的漂移会累积到十几像素，映射回目标帧时已经是错位内容
        # （实测平均误差 26 灰阶、整体偏暗 10 灰阶），补进去就是重影和脏块。
        # 邻近段板之间只隔几十帧，漂移小一个量级，搬过来的才是对的内容。
        span = max(seg, int(seg * self.cfg.donor_span))
        plates: list[tuple[int, np.ndarray]] = []
        hole_ratios: list[float] = []
        for k in range(len(raw)):
            if should_cancel is not None and should_cancel():
                raise CleanError("已取消")
            center, plate, holes = raw[k]
            holes = holes.copy()
            order = sorted(
                (j for j in range(len(raw)) if j != k and abs(raw[j][0] - center) <= span),
                key=lambda j: abs(raw[j][0] - center),
            )
            for j in order:
                if not (holes > 127).any():
                    break
                dc, dp, dh = raw[j]
                plate, holes = self._fill_from_donor(
                    plate, holes, dp, dh, dc, center, homographies,
                    (store.width, store.height),
                )
            ratio = float((holes > 127).mean())
            hole_ratios.append(ratio)
            if self.cfg.inpaint and self.inpainter is not None and self.inpainter.available and (holes > 127).any():
                plate = self.inpainter.fill(plate, holes)
                if on_progress is not None:
                    label = "扩散模型" if self.cfg.inpaint_model == "diffusion" else "LaMa"
                    on_progress("inpaint", k + 1, len(raw), f"{label} 修复背景板 {k + 1}/{len(raw)}")
            plates.append((center, plate))
        return plates, float(np.mean(hole_ratios)) if hole_ratios else 0.0

    def _background_for(
        self,
        frame_idx: int,
        plates: list[tuple[int, np.ndarray]],
        homographies: list[np.ndarray],
        size: tuple[int, int],
    ) -> np.ndarray:
        """取第 ``frame_idx`` 帧对应的背景（相邻两张板按时间权重交叉淡化）。"""
        w, h = size
        centers = [c for c, _ in plates]
        pos = int(np.searchsorted(centers, frame_idx))
        if not self.cfg.crossfade or len(plates) == 1:
            idx = min(max(pos, 0), len(plates) - 1)
            if pos > 0 and abs(centers[pos - 1] - frame_idx) < abs(centers[idx] - frame_idx):
                idx = pos - 1
            center, plate = plates[idx]
            H = _accumulate_homography(homographies, center, frame_idx)
            return cv2.warpPerspective(plate, H, (w, h), flags=cv2.INTER_LINEAR)

        left = min(max(pos - 1, 0), len(plates) - 1)
        right = min(max(pos, 0), len(plates) - 1)
        if left == right:
            center, plate = plates[left]
            H = _accumulate_homography(homographies, center, frame_idx)
            return cv2.warpPerspective(plate, H, (w, h), flags=cv2.INTER_LINEAR)

        c_l, p_l = plates[left]
        c_r, p_r = plates[right]
        span = max(1, c_r - c_l)
        t = float(np.clip((frame_idx - c_l) / span, 0.0, 1.0))
        # smoothstep，避免线性混合在端点出现折线感
        t = t * t * (3 - 2 * t)
        bg_l = cv2.warpPerspective(p_l, _accumulate_homography(homographies, c_l, frame_idx), (w, h),
                                   flags=cv2.INTER_LINEAR).astype(np.float32)
        bg_r = cv2.warpPerspective(p_r, _accumulate_homography(homographies, c_r, frame_idx), (w, h),
                                   flags=cv2.INTER_LINEAR).astype(np.float32)
        return (bg_l * (1 - t) + bg_r * t).astype(np.uint8)

    # ---------------------------------------------------------------- 合成

    def composite(
        self,
        store: FrameStore,
        remove: MaskStore,
        plates: list[tuple[int, np.ndarray]],
        homographies: list[np.ndarray],
        frame_range: tuple[int, int] | None = None,
        on_progress: ProgressFn | None = None,
        should_cancel: Callable[[], bool] | None = None,
    ) -> Iterator[np.ndarray]:
        """逐帧输出「旁人已消失」的画面。区间之外的帧原样输出。"""
        lo = max(0, int(frame_range[0])) if frame_range else 0
        hi = min(store.n_frames, int(frame_range[1])) if frame_range else store.n_frames
        for i in range(store.n_frames):
            if should_cancel is not None and should_cancel():
                raise CleanError("已取消")
            frame = store.frame(i)
            mask = remove.get(i) if lo <= i < hi else np.zeros((1, 1), np.uint8)
            if not mask.any():
                yield frame
            else:
                background = self._background_for(i, plates, homographies, (store.width, store.height))
                alpha = mask.astype(np.float32)
                if self.cfg.feather > 0:
                    alpha = cv2.GaussianBlur(alpha, (0, 0), self.cfg.feather)
                alpha = (alpha / 255.0)[..., None]
                out = frame.astype(np.float32) * (1 - alpha) + background.astype(np.float32) * alpha
                yield out.astype(np.uint8)
            if on_progress is not None and (i % 10 == 0 or i == store.n_frames - 1):
                on_progress("compose", i + 1, store.n_frames, f"合成中 {i + 1}/{store.n_frames}")


def remove_people(
    src_video: str | Path,
    out_path: str | Path,
    target_masks: MaskStore,
    models_dir: str | Path,
    work_dir: str | Path,
    cfg: CleanConfig | None = None,
    frame_range: tuple[int, int] | None = None,
    on_progress: ProgressFn | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict:
    """端到端：把除目标外的所有人从视频里抹掉。

    ``target_masks`` 是 SAM2 在工作分辨率下算出来的目标掩码，这里会放大到源分辨率
    再和「全体人员」相减。输出是一段干净的视频（ffv1 无损，供后续虚化流程继续处理，
    避免二次压缩损失）。
    """
    cfg = cfg or CleanConfig()
    started = time.perf_counter()
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    inpainter = None
    if cfg.inpaint:
        if cfg.inpaint_model == "diffusion":
            from .inpaint_diffusion import DiffusionInpainter, resolve_diffusion

            dmodel = resolve_diffusion(models_dir)
            if dmodel is None:
                raise CleanError(
                    "未找到扩散补洞模型，请先执行：python -m spotlight.cli download --preset diffusion"
                )
            inpainter = DiffusionInpainter(dmodel)
            if not inpainter.available:
                raise CleanError(f"扩散补洞模型加载失败：{inpainter.reason}")
        else:
            model = resolve_lama(models_dir)
            if model is not None:
                inpainter = LamaInpainter(model)
    engine = None
    try:
        from .appearance import AppearanceEngine

        engine = AppearanceEngine()
    except Exception:  # noqa: BLE001 - 没有 Vision 就不做人体框兜底
        engine = None
    remover = PeopleRemover(cfg, inpainter, engine)

    # 在**源分辨率**上做，避免整段画面被降分辨率拖糊
    store = FrameStore(src_video, work / "clean_frames", long_side=None)
    try:
        n, w, h = store.n_frames, store.width, store.height
        if on_progress is not None:
            on_progress("clean", 0, n, f"清除旁人：源分辨率 {w}x{h}，共 {n} 帧")

        # 目标掩码放大到源分辨率
        targets = MaskStore(work / "targets_src.u8", n, w, h)
        for i in range(n):
            if should_cancel is not None and should_cancel():
                raise CleanError("已取消")
            m = target_masks.get(min(i, target_masks.n - 1))
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
            targets.set(i, m)

        persons = remover.build_person_masks(store, work, on_progress, should_cancel)
        remove = remover.build_remove_masks(store, persons, targets, work, on_progress, should_cancel)

        # 要抹掉的像素太少就直接跳过，省得白跑一遍重建
        rng = frame_range or (0, n)
        idx = list(range(rng[0], rng[1], max(1, (rng[1] - rng[0]) // 20))) or [0]
        coverage = float(np.mean([int((remove.get(i) > 127).sum()) for i in idx])) / float(w * h)
        if coverage < 0.0005:
            if on_progress is not None:
                on_progress("clean", n, n, "画面里没有需要清除的旁人")
            write_frame_video(store.iter_frames(), out_path, w, h, store.info.fps_expr,
                              total=n, encoder="ffv1")
            return {"coverage": 0.0, "hole_ratio": 0.0, "segments": 0, "inpainted": False,
                    "elapsed": time.perf_counter() - started, "skipped": True}

        homographies = _frame_homographies(store, persons, cfg, on_progress, should_cancel)
        plates, hole_ratio = remover.build_plates(
            store, persons, homographies, frame_range=frame_range,
            on_progress=on_progress, should_cancel=should_cancel,
        )

        write_frame_video(
            remover.composite(store, remove, plates, homographies, frame_range=frame_range,
                              on_progress=on_progress, should_cancel=should_cancel),
            out_path, w, h, store.info.fps_expr, total=n, encoder="ffv1",
        )

        stats = {
            "coverage": round(coverage, 4),
            "hole_ratio": round(hole_ratio, 4),
            "segments": len(plates),
            "inpainted": bool(inpainter is not None and inpainter.available),
            "inpaint_model": cfg.inpaint_model,
            "elapsed": round(time.perf_counter() - started, 2),
            "skipped": False,
        }
        if on_progress is not None:
            on_progress("clean", n, n, f"旁人已清除（覆盖 {coverage:.1%}，空洞 {hole_ratio:.1%}）", **stats)
        return stats
    finally:
        store.close()

