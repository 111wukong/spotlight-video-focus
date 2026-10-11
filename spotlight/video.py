"""视频探测 / 解码 / 帧存储 / 编码，全部通过 ffmpeg 完成。

设计要点
--------
* 帧不放在内存里。解码结果写成一个磁盘上的 ``uint8 RGB`` memmap，
  推理时按帧随机读取，内存占用与视频长度无关。
* 解码时强制 CFR（``-r <fps>``），这样「原始视频」和「掩码视频」的帧
  一一对应，合成时不需要再做对轴。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


class VideoError(RuntimeError):
    pass


def _run(cmd: list[str], stdin=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _parse_rate(expr: str | None) -> float:
    if not expr:
        return 0.0
    try:
        f = Fraction(expr)
    except (ValueError, ZeroDivisionError):
        return 0.0
    return float(f) if f > 0 else 0.0


def _even(n: int) -> int:
    n = int(round(n))
    return n if n % 2 == 0 else n - 1


@dataclass
class VideoInfo:
    path: str
    width: int
    height: int
    fps: float
    fps_expr: str
    n_frames: int
    duration: float
    has_audio: bool
    codec: str
    rotation: int = 0

    @property
    def resolution(self) -> tuple[int, int]:
        return self.width, self.height

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4),
            "fps_expr": self.fps_expr,
            "n_frames": self.n_frames,
            "duration": round(self.duration, 3),
            "has_audio": self.has_audio,
            "codec": self.codec,
        }


def probe(path: str | Path) -> VideoInfo:
    path = str(path)
    out = _run(
        [FFPROBE, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", path]
    )
    if out.returncode != 0:
        raise VideoError(f"ffprobe 读取失败：{out.stderr.decode(errors='replace')[:400]}")
    try:
        data = json.loads(out.stdout.decode("utf-8", "replace") or "{}")
    except json.JSONDecodeError as exc:  # pragma: no cover
        raise VideoError(f"ffprobe 输出无法解析：{exc}") from exc

    streams = data.get("streams") or []
    vs = next((s for s in streams if s.get("codec_type") == "video"), None)
    if vs is None:
        raise VideoError("这个文件里没有视频流")

    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    fps_expr = vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "25/1"
    fps = _parse_rate(fps_expr)
    if fps <= 0:
        fps_expr, fps = "25/1", 25.0

    width = int(vs.get("width") or 0)
    height = int(vs.get("height") or 0)

    rotation = 0
    for sd in vs.get("side_data_list") or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])

    duration = 0.0
    for src in (vs.get("duration"), (data.get("format") or {}).get("duration")):
        try:
            duration = float(src)
            break
        except (TypeError, ValueError):
            continue

    n_frames = 0
    try:
        n_frames = int(vs.get("nb_frames") or 0)
    except (TypeError, ValueError):
        n_frames = 0
    if n_frames <= 0:
        n_frames = max(1, int(round(duration * fps))) if duration else 0

    # 旋转信息由 ffmpeg 解码时自动应用，宽高按显示方向归一
    if abs(rotation) % 180 == 90:
        width, height = height, width

    return VideoInfo(
        path=path,
        width=width,
        height=height,
        fps=fps,
        fps_expr=fps_expr,
        n_frames=n_frames,
        duration=duration or (n_frames / fps if fps else 0.0),
        has_audio=has_audio,
        codec=str(vs.get("codec_name") or "?"),
        rotation=rotation,
    )


class FrameStore:
    """把视频解码成磁盘上的 uint8 RGB memmap，随机访问且不常驻内存。"""

    def __init__(
        self,
        path: str | Path,
        work_dir: str | Path,
        long_side: int | None = 1280,
        info: VideoInfo | None = None,
    ) -> None:
        self.info = info or probe(path)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        w, h = self.info.width, self.info.height
        if long_side and max(w, h) > long_side:
            s = long_side / max(w, h)
            w, h = _even(round(w * s)), _even(round(h * s))
        self.width, self.height = max(2, w), max(2, h)

        self.raw_path = self.work_dir / "frames.rgb"
        self.n_frames = self._decode()
        self._mm = np.memmap(
            self.raw_path, dtype=np.uint8, mode="r", shape=(self.n_frames, self.height, self.width, 3)
        )

    @property
    def scaled(self) -> bool:
        """是否为了省算力把画面缩小过。"""
        return (self.width, self.height) != (self.info.width, self.info.height)

    def _decode(self) -> int:
        vf = f"scale={self.width}:{self.height}:flags=bicubic"
        cmd = [
            FFMPEG, "-v", "error", "-nostdin",
            "-i", self.info.path,
            "-vf", vf,
            "-r", self.info.fps_expr,
            "-fps_mode", "cfr",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-y", str(self.raw_path),
        ]
        proc = _run(cmd)
        if proc.returncode != 0:
            raise VideoError(
                f"ffmpeg 解码失败：{proc.stderr.decode(errors='replace')[:400]}"
            )
        per = self.width * self.height * 3
        size = self.raw_path.stat().st_size
        n = size // per
        if n <= 0:
            raise VideoError("没有解出任何帧，文件可能损坏或不是视频")
        if size % per:
            with open(self.raw_path, "r+b") as fh:
                fh.truncate(n * per)
        return n

    def frame(self, idx: int) -> np.ndarray:
        """返回第 idx 帧的 RGB uint8 视图（只读）。"""
        idx = max(0, min(self.n_frames - 1, int(idx)))
        return self._mm[idx]

    def __len__(self) -> int:
        return self.n_frames

    def close(self) -> None:
        mm = getattr(self, "_mm", None)
        if mm is not None:
            mm._mmap.close()  # type: ignore[attr-defined]
            self._mm = None  # type: ignore[assignment]

    def __enter__(self) -> "FrameStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class MaskStore:
    """掩码的磁盘存储。

    为什么不能放内存：1280x720 的 uint8 掩码每帧 0.92 MB，
    300 秒 30fps 的素材就是 8.3 GB —— 而且补齐空洞、跳帧插值还会各再复制一份。

    为什么用 ``pwrite``/``pread`` 而不是 ``np.memmap``：写 memmap 产生的**脏页
    会计入进程 RSS**（实测写 11 GB 掩码，RSS 涨到 8.8 GB），在 24 GB 的机器上
    依然很危险。走普通文件 I/O 时数据落在内核页缓存里，不计入进程内存，
    实测同样的写入量 RSS 基本不动。

    ``written`` 记录哪些帧是推理真算出来的（其余是需要补齐的空洞）。
    """

    def __init__(self, path: str | Path, n_frames: int, width: int, height: int) -> None:
        self.path = Path(path)
        self.n = int(n_frames)
        self.w = int(width)
        self.h = int(height)
        self.frame_bytes = self.w * self.h
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
        os.ftruncate(self._fd, self.n * self.frame_bytes)
        self.written = np.zeros(self.n, dtype=bool)

    # ---------------------------------------------------------------- 读写

    def set(self, idx: int, mask: np.ndarray) -> None:
        if mask.shape != (self.h, self.w):
            raise ValueError(f"掩码尺寸不符：期望 {(self.h, self.w)}，得到 {mask.shape}")
        buf = np.ascontiguousarray(mask, dtype=np.uint8).tobytes()
        os.pwrite(self._fd, buf, int(idx) * self.frame_bytes)
        self.written[int(idx)] = True

    def _read(self, idx: int) -> np.ndarray:
        buf = os.pread(self._fd, self.frame_bytes, int(idx) * self.frame_bytes)
        return np.frombuffer(buf, dtype=np.uint8).reshape(self.h, self.w)

    def get(self, idx: int) -> np.ndarray:
        return self._read(idx)

    def has(self, idx: int) -> bool:
        return bool(self.written[idx])

    @property
    def n_written(self) -> int:
        return int(self.written.sum())

    # ---------------------------------------------------------------- 补齐

    def finalize(self, interpolate: bool = True) -> dict:
        """把没被推理覆盖到的帧补齐。

        推理范围之外（片头/片尾）用最近的有效掩码保持；
        中间的空洞（跳帧推理留下的）在两侧掩码之间做线性过渡，避免台阶感。
        """
        idxs = np.flatnonzero(self.written)
        if idxs.size == 0:
            return {"held": 0, "interpolated": 0}

        first, last = int(idxs[0]), int(idxs[-1])
        held = 0
        if first > 0:
            self._fill(first, first, 0)
            held += first
        if last < self.n - 1:
            self._fill(last, last, last + 1)
            held += self.n - 1 - last

        interpolated = 0
        for a, b in zip(idxs[:-1], idxs[1:]):
            gap = int(b) - int(a) - 1
            if gap <= 0:
                continue
            if not interpolate:
                self._fill(int(a), int(a), int(a) + 1)
                held += gap
                continue
            lo = self._read(int(a)).astype(np.uint16)
            hi = self._read(int(b)).astype(np.uint16)
            span = int(b) - int(a)
            for i in range(1, span):
                t = i / span
                self._write(int(a) + i, ((1.0 - t) * lo + t * hi).astype(np.uint8))
            interpolated += gap
        return {"held": held, "interpolated": interpolated}

    def _write(self, idx: int, arr: np.ndarray) -> None:
        os.pwrite(self._fd, np.ascontiguousarray(arr, dtype=np.uint8).tobytes(), int(idx) * self.frame_bytes)

    def _fill(self, src: int, start: int, end: int) -> None:
        """把 ``[start, end)`` 区间用第 ``src`` 帧的掩码填充。"""
        if end <= start:
            return
        src_arr = self._read(src)
        buf = np.ascontiguousarray(src_arr, dtype=np.uint8).tobytes()
        for i in range(int(start), int(end)):
            os.pwrite(self._fd, buf, i * self.frame_bytes)

    def clear(self, indices) -> None:
        """把指定的帧清成空掩码（用于「只处理某一段」时把范围外抹掉）。"""
        blank = np.zeros(self.frame_bytes, dtype=np.uint8).tobytes()
        for i in np.asarray(indices).reshape(-1):
            os.pwrite(self._fd, blank, int(i) * self.frame_bytes)

    # ---------------------------------------------------------------- 输出

    def iter_frames(self):
        for i in range(self.n):
            yield self._read(i)

    def scan(self, min_pixels: int) -> dict:
        """一趟扫完，同时得到逐帧「目标是否存在」和掩码面积。

        掩码在磁盘上，每帧读一次 0.92 MB，所以统计指标要合并成一趟算，
        不能为了「面积分布」和「丢失片段」各扫一遍。
        """
        present = np.zeros(self.n, dtype=bool)
        areas = np.zeros(self.n, dtype=np.int64)
        for i in range(self.n):
            area = int((self._read(i) > 60).sum())
            areas[i] = area
            present[i] = area >= min_pixels
        return {"present": present, "areas": areas}

    def close(self) -> None:
        fd = getattr(self, "_fd", None)
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self) -> "MaskStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def gate_levels(present: np.ndarray, ramp_up: int = 3, ramp_down: int = 7) -> np.ndarray:
    """逐帧「目标是否存在」的 0~255 电平，带渐变。

    做成渐变而不是硬切，是为了让目标走进/走出画面时画面平滑过渡，
    不会「啪」地一下从全虚化跳到全清晰。信号是因果的（只看当前帧和上一帧）。
    """
    levels = np.zeros(len(present), dtype=np.uint8)
    step_up = 1.0 / max(1, ramp_up)
    step_down = 1.0 / max(1, ramp_down)
    level = 0.0
    for i, flag in enumerate(present):
        target = 1.0 if flag else 0.0
        level = min(target, level + step_up) if target > level else max(target, level - step_down)
        levels[i] = int(round(level * 255))
    return levels


def lost_segments(present: np.ndarray, min_len: int = 3) -> list[dict]:
    """找出目标「不在画面里」的连续片段，用于告诉用户哪里没跟上。"""
    segs: list[dict] = []
    n = len(present)
    i = 0
    while i < n:
        if not present[i]:
            j = i
            while j < n and not present[j]:
                j += 1
            if j - i >= min_len:
                segs.append({"start": i, "end": j - 1, "frames": j - i})
            i = j
        else:
            i += 1
    return segs


def summarize_masks(areas: np.ndarray, present: np.ndarray) -> dict:
    """掩码覆盖率的分位数，给用户一个「目标占画面多大」的直观量。"""
    if not present.any():
        return {"frames_with_mask": 0, "area_median": 0, "area_p10": 0, "area_p90": 0}
    vals = areas[present]
    return {
        "frames_with_mask": int(present.sum()),
        "area_median": int(np.median(vals)),
        "area_p10": int(np.percentile(vals, 10)),
        "area_p90": int(np.percentile(vals, 90)),
    }


def write_frame_video(
    frames,
    out_path: str | Path,
    width: int,
    height: int,
    fps_expr: str,
    total: int | None = None,
    encoder: str = "libx264",
    quality: int = 14,
) -> int:
    """把逐帧的 ``uint8 RGB`` 画面写成视频。

    ``encoder`` 传 ``ffv1`` 时写无损中间文件（供第二阶段继续处理，避免二次压缩损失）。
    """
    out_path = str(out_path)
    codec = ["-c:v", "ffv1", "-level", "3"] if encoder == "ffv1" else [
        "-c:v", "libx264", "-crf", str(quality), "-preset", "veryfast", "-pix_fmt", "yuv420p",
    ]
    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", fps_expr,
        "-i", "-",
        *codec, "-y", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    count = 0
    try:
        for frame in frames:
            if frame.shape[:2] != (height, width):
                raise ValueError(f"帧尺寸不符：期望 {(height, width)}，得到 {frame.shape[:2]}")
            proc.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            count += 1
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read() if proc.stderr else b""
        proc.wait()
    if proc.returncode != 0:
        raise VideoError(f"写入视频失败：{err.decode(errors='replace')[:400]}")
    if total is not None and count != total:
        raise VideoError(f"帧数不匹配：期望 {total}，实际 {count}")
    return count


def write_mask_video(    masks,
    out_path: str | Path,
    width: int,
    height: int,
    fps_expr: str,
    total: int | None = None,
) -> int:
    """把逐帧的 uint8 灰度掩码写成无损视频（ffv1/mkv）。

    ``masks`` 是可迭代对象，每项为 ``(h, w)`` 的 uint8 数组。
    """
    out_path = str(out_path)
    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{width}x{height}", "-r", fps_expr,
        "-i", "-",
        "-c:v", "ffv1", "-level", "3", "-g", "1",
        "-y", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    count = 0
    try:
        for m in masks:
            if m.shape != (height, width):
                raise ValueError(f"掩码尺寸不符：期望 {(height, width)}，得到 {m.shape}")
            proc.stdin.write(np.ascontiguousarray(m, dtype=np.uint8).tobytes())
            count += 1
    finally:
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read() if proc.stderr else b""
        proc.wait()
    if proc.returncode != 0:
        raise VideoError(f"写入掩码视频失败：{err.decode(errors='replace')[:400]}")
    if total is not None and count != total:
        raise VideoError(f"掩码帧数不匹配：期望 {total}，实际 {count}")
    return count


def encode_output(
    src_video: str | Path,
    mask_video: str | Path,
    out_path: str | Path,
    filtergraph: str,
    info: VideoInfo,
    encoder: str = "auto",
    quality: int = 18,
    audio: bool = True,
) -> None:
    """用一条 ffmpeg 命令完成「虚化 + 合成 + 编码」。"""
    if encoder == "auto":
        encoder = "h264_videotoolbox" if shutil.which("ffmpeg") else "libx264"

    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-i", str(src_video),
        "-i", str(mask_video),
        "-filter_complex", filtergraph,
        "-map", "[vout]",
    ]
    if info.has_audio and audio:
        cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]

    if encoder == "h264_videotoolbox":
        cmd += ["-c:v", "h264_videotoolbox", "-q:v", str(max(1, min(100, quality)))]
    elif encoder == "hevc_videotoolbox":
        cmd += ["-c:v", "hevc_videotoolbox", "-q:v", str(max(1, min(100, quality))), "-tag:v", "hvc1"]
    else:
        cmd += ["-c:v", encoder, "-crf", str(quality), "-preset", "veryfast"]

    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(out_path)]

    proc = _run(cmd)
    if proc.returncode != 0:
        raise VideoError(f"合成编码失败：{proc.stderr.decode(errors='replace')[:900]}")


def extract_jpeg(video_path: str | Path, frame_idx: int, fps: float, out_path: str | Path, width: int = 1280) -> None:
    """抽取单帧为 JPEG（给网页端选人用）。"""
    t = max(0.0, frame_idx / fps) if fps else 0.0
    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-ss", f"{t:.6f}", "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale={width}:-2:flags=bicubic",
        "-q:v", "3", "-y", str(out_path),
    ]
    proc = _run(cmd)
    if proc.returncode != 0 or not Path(out_path).exists():
        raise VideoError(f"抽帧失败：{proc.stderr.decode(errors='replace')[:300]}")


@dataclass
class Cleanup:
    """临时目录集合，退出时统一删除。"""

    paths: list[Path] = field(default_factory=list)

    def add(self, p: str | Path) -> Path:
        p = Path(p)
        self.paths.append(p)
        return p

    def run(self) -> None:
        for p in self.paths:
            shutil.rmtree(p, ignore_errors=True)
        self.paths.clear()
