"""目标外观表征：人体检测 + 外观特征（用于「丢失后重新找回同一个人」）。

全部基于系统内置的 Vision 框架，模型随系统提供、跑在 ANE 上，不需要额外下载，
也不需要 PyTorch：

* :class:`~AppearanceEngine.detect_people` 用 ``VNDetectHumanRectanglesRequest``
  找出画面里所有可能的人体框——这是目标丢失后的「候选池」。
* :class:`~AppearanceEngine.embed` 用 ``VNGenerateImageFeaturePrintRequest``
  对候选裁剪算 768 维特征向量，与目标模板比对——这是「不换人」的判据。

若运行环境拿不到 Vision（非 macOS 等），会自动降级为纯颜色直方图方案，
功能仍在但重识别能力会弱一些。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

try:  # pragma: no cover - 平台相关
    import Quartz
    import Vision

    _VISION_OK = True
except Exception:  # pragma: no cover
    Quartz = None  # type: ignore[assignment]
    Vision = None  # type: ignore[assignment]
    _VISION_OK = False


@dataclass(frozen=True)
class PersonBox:
    """像素坐标（左上原点）的人体框。"""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def expand(self, ratio: float, w: int, h: int) -> "PersonBox":
        dx, dy = self.width * ratio, self.height * ratio
        return PersonBox(
            max(0.0, self.x1 - dx),
            max(0.0, self.y1 - dy),
            min(float(w), self.x2 + dx),
            min(float(h), self.y2 + dy),
            self.confidence,
        )


def _vision_rect_to_box(bb, w: int, h: int, conf: float) -> PersonBox | None:
    """Vision 的归一化矩形（原点左下、y 向上）→ 像素框（原点左上、y 向下）。"""
    x1 = float(bb.origin.x) * w
    x2 = (float(bb.origin.x) + float(bb.size.width)) * w
    y1 = (1.0 - float(bb.origin.y) - float(bb.size.height)) * h
    y2 = (1.0 - float(bb.origin.y)) * h
    x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
    if (x2 - x1) < 8 or (y2 - y1) < 8:
        return None
    return PersonBox(x1, y1, x2, y2, conf)


def _dedupe(boxes: Sequence[PersonBox], iou_threshold: float = 0.45) -> list[PersonBox]:
    """按置信度去重：重叠度高的只留最可信的那个。"""
    kept: list[PersonBox] = []
    for box in sorted(boxes, key=lambda b: -b.confidence):
        if all(_box_iou(box, other) < iou_threshold for other in kept):
            kept.append(box)
    return kept


def _box_iou(a: PersonBox, b: PersonBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


class PersonSegmenter:
    """逐帧分割出画面里**所有**人（不是某一个人）。

    用的是系统内置的 ``VNGeneratePersonSegmentationRequest``——它本来是给视频通话
    背景替换用的，跑在 ANE 上，热态约 35 ms/帧，不需要下载任何模型。

    和 SAM2 的分工：SAM2 负责「精确锁定某一个人」（要跨帧保持身份），
    这里负责「把所有人一网打尽」（只需要当帧的覆盖面）。
    两者相减就是「要抹掉的人」。
    """

    def __init__(self, quality: str = "accurate") -> None:
        self.available = _VISION_OK
        self._quality = {
            "accurate": getattr(Vision, "VNGeneratePersonSegmentationRequestQualityLevelAccurate", 2),
            "balanced": getattr(Vision, "VNGeneratePersonSegmentationRequestQualityLevelBalanced", 1),
            "fast": getattr(Vision, "VNGeneratePersonSegmentationRequestQualityLevelFast", 0),
        }.get(quality, 2)

    def mask(self, rgb: np.ndarray, out_size: tuple[int, int] | None = None) -> np.ndarray | None:
        """返回 ``uint8 HxW`` 的人员掩码（255=人，0=背景）。失败返回 None。"""
        if not _VISION_OK:
            return None
        cg = _numpy_to_cgimage(rgb)
        if cg is None:
            return None
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, {})
        request = Vision.VNGeneratePersonSegmentationRequest.alloc().initWithCompletionHandler_(None)
        try:
            request.setQualityLevel_(self._quality)
        except Exception:
            pass
        ok, _err = handler.performRequests_error_([request], None)
        if not ok:
            return None
        results = request.results() or []
        if not results:
            return None
        buffer = results[0].pixelBuffer()
        if buffer is None:
            return None

        width = Quartz.CVPixelBufferGetWidth(buffer)
        height = Quartz.CVPixelBufferGetHeight(buffer)
        Quartz.CVPixelBufferLockBaseAddress(buffer, 1)  # kCVPixelBufferLock_ReadOnly
        try:
            base = Quartz.CVPixelBufferGetBaseAddress(buffer)
            stride = Quartz.CVPixelBufferGetBytesPerRow(buffer)
            raw = np.frombuffer(base.as_buffer(stride * height), dtype=np.uint8)
            plane = raw.reshape(height, stride)[:, :width].copy()
        finally:
            Quartz.CVPixelBufferUnlockBaseAddress(buffer, 1)

        if out_size is not None and (width, height) != (out_size[0], out_size[1]):
            plane = cv2.resize(plane, out_size, interpolation=cv2.INTER_LINEAR)
        return plane


def _numpy_to_cgimage(rgb: np.ndarray):
    """``uint8 HWC RGB`` → ``CGImage``。"""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w = rgb.shape[:2]
    buf = rgb.tobytes()
    provider = Quartz.CGDataProviderCreateWithData(None, buf, len(buf), None)
    colorspace = Quartz.CGColorSpaceCreateDeviceRGB()
    return Quartz.CGImageCreate(
        w, h, 8, 24, w * 3, colorspace,
        Quartz.kCGBitmapByteOrderDefault | Quartz.kCGImageAlphaNone,
        provider, None, False, Quartz.kCGRenderingIntentDefault,
    )


class AppearanceEngine:
    """人体检测 + 外观特征提取。"""

    def __init__(self, hist_bins: int = 16) -> None:
        self.hist_bins = int(hist_bins)
        self.vision_available = _VISION_OK
        self._detect_warm = False

    # ------------------------------------------------------------ 人体检测

    def detect_people(
        self,
        rgb: np.ndarray,
        min_confidence: float = 0.12,
        upper_body_only: bool = False,
    ) -> list[PersonBox]:
        """返回画面中所有可能是人的框（像素坐标，左上原点）。

        这里刻意把阈值放得很低，并**合并三种检测器**（人体矩形 / 人脸 / 人体姿态）。
        原因是这些框只作为「候选池」，真正决定要不要锁定的是后面的外观比对——
        对候选池来说召回率比准确率重要得多，漏掉真目标才是致命伤。
        （实测：单一人体矩形检测在密集人群里只给出 1 个高置信框，三者合并后有 8~10 个。）
        """
        if not _VISION_OK:
            return []
        h, w = rgb.shape[:2]
        cg = _numpy_to_cgimage(rgb)
        if cg is None:
            return []
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, {})

        boxes: list[PersonBox] = []
        boxes += self._human_rectangles(handler, w, h, min_confidence, upper_body_only)
        boxes += self._faces_as_bodies(handler, w, h, min_confidence)
        boxes += self._poses_as_bodies(handler, w, h)
        self._detect_warm = True
        return _dedupe(boxes)

    @staticmethod
    def _perform(handler, request) -> list:
        ok, _err = handler.performRequests_error_([request], None)
        return list(request.results() or []) if ok else []

    def _human_rectangles(self, handler, w: int, h: int, min_conf: float, upper_body_only: bool) -> list[PersonBox]:
        request = Vision.VNDetectHumanRectanglesRequest.alloc().init()
        try:
            request.setUpperBodyOnly_(bool(upper_body_only))
        except Exception:
            pass
        out = []
        for obs in self._perform(handler, request):
            conf = float(obs.confidence())
            if conf < min_conf:
                continue
            box = _vision_rect_to_box(obs.boundingBox(), w, h, conf)
            if box is not None:
                out.append(box)
        return out

    def _faces_as_bodies(self, handler, w: int, h: int, min_conf: float) -> list[PersonBox]:
        """人脸框 → 估算的身体框。

        人脸检测的召回率明显高于人体矩形检测，在人群里是很重要的补充来源。
        用一个粗略的人体比例把脸框外扩成身体框：肩宽约 3 倍脸宽，身高约 7.5 个头。
        """
        request = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        out = []
        for obs in self._perform(handler, request):
            conf = float(obs.confidence())
            if conf < min_conf:
                continue
            face = _vision_rect_to_box(obs.boundingBox(), w, h, conf)
            if face is None:
                continue
            cx = (face.x1 + face.x2) / 2.0
            half_w = face.width * 1.55
            out.append(PersonBox(
                max(0.0, cx - half_w),
                max(0.0, face.y1 - face.height * 0.25),
                min(float(w), cx + half_w),
                min(float(h), face.y1 + face.height * 7.4),
                conf * 0.9,
            ))
        return out

    def _poses_as_bodies(self, handler, w: int, h: int, joint_conf: float = 0.25) -> list[PersonBox]:
        """人体姿态的关键点 → 外接框。背对镜头、没有正脸时的主要来源。"""
        request = Vision.VNDetectHumanBodyPoseRequest.alloc().init()
        out = []
        for obs in self._perform(handler, request):
            try:
                points, _err = obs.recognizedPointsForGroupKey_error_(
                    Vision.VNHumanBodyPoseObservationJointsGroupNameAll, None
                )
            except Exception:
                continue
            xs, ys, confs = [], [], []
            for point in (points or {}).values():
                c = float(point.confidence())
                if c < joint_conf:
                    continue
                loc = point.location()
                xs.append(float(loc.x) * w)
                ys.append((1.0 - float(loc.y)) * h)
                confs.append(c)
            if len(xs) < 4:
                continue
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            pad_x = (x2 - x1) * 0.35 + 8
            pad_y = (y2 - y1) * 0.15 + 8
            out.append(PersonBox(
                max(0.0, x1 - pad_x), max(0.0, y1 - pad_y),
                min(float(w), x2 + pad_x), min(float(h), y2 + pad_y),
                float(np.mean(confs)),
            ))
        return out

    # ------------------------------------------------------------ 外观特征

    def feature_print(self, rgb: np.ndarray) -> np.ndarray | None:
        """整幅图的 768 维外观特征（已 L2 归一化）。"""
        if not _VISION_OK:
            return None
        cg = _numpy_to_cgimage(rgb)
        if cg is None:
            return None
        handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, {})
        request = Vision.VNGenerateImageFeaturePrintRequest.alloc().init()
        ok, _err = handler.performRequests_error_([request], None)
        if not ok:
            return None
        results = request.results() or []
        if not results:
            return None
        data = results[0].data()
        if data is None:
            return None
        vec = np.frombuffer(bytes(data), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm > 1e-6 else None

    def embed(self, rgb: np.ndarray, box: PersonBox, pad: float = 0.06) -> np.ndarray | None:
        """对某个框内的区域算外观特征。"""
        h, w = rgb.shape[:2]
        b = box.expand(pad, w, h) if pad else box
        x1, y1 = int(max(0, np.floor(b.x1))), int(max(0, np.floor(b.y1)))
        x2, y2 = int(min(w, np.ceil(b.x2))), int(min(h, np.ceil(b.y2)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        crop = rgb[y1:y2, x1:x2]
        return self.feature_print(crop)

    def embed_masked(self, rgb: np.ndarray, mask: np.ndarray, pad: float = 0.05) -> np.ndarray | None:
        """对掩码覆盖的区域算外观特征（先裁到掩码外接框）。"""
        box = mask_to_box(mask)
        if box is None:
            return None
        return self.embed(rgb, box, pad=pad)

    # ------------------------------------------------------------ 颜色直方图

    def color_hist(self, rgb: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """HSV 二维直方图（H×S），作为特征向量之外的补充线索。"""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        bins = self.hist_bins
        if mask is None:
            hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
        else:
            m = (mask > 0).astype(np.uint8) * 255
            if m.sum() == 0:
                return np.zeros(bins * bins, dtype=np.float32)
            hist = cv2.calcHist([hsv], [0, 1], m, [bins, bins], [0, 180, 0, 256])
        hist = hist.flatten().astype(np.float32)
        total = float(hist.sum())
        return hist / total if total > 1e-6 else hist


def mask_to_box(mask: np.ndarray, threshold: int = 127) -> PersonBox | None:
    """掩码的外接框。"""
    ys, xs = np.nonzero(mask > threshold)
    if xs.size == 0:
        return None
    return PersonBox(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def mask_centroid(mask: np.ndarray, threshold: int = 127) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask > threshold)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def box_iou(a: PersonBox, b: PersonBox) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
    """直方图相似度，1 表示完全一致。"""
    if a.size == 0 or b.size == 0:
        return 0.0
    bc = float(np.sum(np.sqrt(np.clip(a, 0, None) * np.clip(b, 0, None))))
    return float(np.clip(bc, 0.0, 1.0))


def best_match(
    template_fp: np.ndarray | None,
    template_hist: np.ndarray | None,
    candidates: Sequence[tuple[np.ndarray | None, np.ndarray]],
    fp_weight: float = 0.65,
) -> tuple[int, float]:
    """在候选里找与模板最像的一个，返回 ``(下标, 综合得分)``。"""
    best_idx, best_score = -1, -1.0
    for idx, (fp, hist) in enumerate(candidates):
        fp_sim = cosine(template_fp, fp) if (template_fp is not None and fp is not None) else None
        hist_sim = bhattacharyya(template_hist, hist) if template_hist is not None else None
        if fp_sim is None and hist_sim is None:
            continue
        if fp_sim is None:
            score = float(hist_sim)
        elif hist_sim is None:
            score = float(fp_sim)
        else:
            score = fp_weight * float(fp_sim) + (1.0 - fp_weight) * float(hist_sim)
        if score > best_score:
            best_idx, best_score = idx, score
    return best_idx, best_score
