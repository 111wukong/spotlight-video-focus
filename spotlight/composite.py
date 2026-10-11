"""合成渲染：背景虚化 + 主体提亮/柔光，全部交给 ffmpeg 的 filtergraph 一次完成。

为什么走 ffmpeg 而不是 numpy
--------------------------
逐帧用 numpy 做高斯模糊再写回视频，1080p 只能跑到个位数 fps，而且要把
整个视频读进内存。ffmpeg 的 ``gblur`` 是 SIMD 优化过的，配合
``h264_videotoolbox`` 硬件编码，整条链路是流式的、内存恒定。

关于「目标不在画面里」的处理
--------------------------
如果只是简单地把「非目标区域」虚化，那么目标走出画面时整帧都会被虚化，
看起来像出错了。所以这里额外引入一路 **存在门（presence gate）**：

* 目标在画面里 → 背景虚化、主体提亮；
* 目标不在画面里 → 原样输出，不做任何处理。

门信号是逐帧的标量，用一路灰度视频承载，配合 ``maskedmerge`` 实现。
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .video import FFMPEG, VideoError, VideoInfo, _run


@dataclass
class EffectParams:
    """视觉参数。默认值对应「虚化 + 提亮主体」这一档。"""

    #: 背景虚化强度（高斯 sigma，越大越虚）
    blur: float = 26.0
    #: 背景亮度调整（-1..1）
    bg_brightness: float = -0.02
    #: 背景饱和度
    bg_saturation: float = 0.88
    #: 主体亮度提升
    fg_brightness: float = 0.055
    #: 主体对比度
    fg_contrast: float = 1.07
    #: 主体饱和度
    fg_saturation: float = 1.04
    #: 主体锐化强度（0 关闭）
    fg_sharpen: float = 0.0
    #: 柔光半径
    glow_sigma: float = 14.0
    #: 柔光叠加强度（0 关闭）
    glow_opacity: float = 0.22
    #: 掩码羽化半径，决定边缘过渡的柔和程度
    feather: float = 3.0
    #: 掩码时域平滑，抑制边缘抖动
    temporal_smooth: bool = True
    #: 掩码形态学闭运算，补掉人物内部的小洞
    fill_holes: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".")


#: 景深虚化的模糊层级数。层越多过渡越细，但每层都要多跑一路 gblur。
#: 4 层（1 路清晰 + 3 路渐强模糊）在观感和开销之间比较平衡。
DEPTH_LEVELS = 4


def _depth_blur_chain(p: EffectParams, w: int, h: int,
                      levels: int = DEPTH_LEVELS) -> tuple[list[str], str]:
    """深度分层虚化的 filtergraph 片段。

    思路：把背景同时模糊成 ``levels`` 个强度（第 0 级就是原图），
    再用**逐像素的层级掩码**把它们叠回去——掩码来自深度视频（``[3:v]``），
    所以近处取清晰那一路、远处取最糊那一路，中间平滑过渡。

    为什么不做「按像素变 sigma」：ffmpeg 的 ``gblur`` 的 sigma 是整帧常量。
    分层 + 逐像素掩码是流式的、内存恒定，而且能复用硬件编码那条链路。

    深度视频里存的是 **模糊层级 0..255**（不是原始深度）——映射在 Python 侧
    算好（见 :func:`spotlight.depth.to_blur_level`），ffmpeg 这边只做阈值与斜坡。
    """
    n_blur = max(1, int(levels) - 1)
    # 每一级的 sigma：从 0 递增到 blur
    sigmas = [p.blur * (k / n_blur) ** 1.15 for k in range(1, n_blur + 1)]
    # 阈值与过渡宽度（0..255 的层级刻度）
    step = 255.0 / (n_blur + 1)
    soft = max(8.0, step * 0.9)

    parts: list[str] = []
    tags = [f"b{i}" for i in range(n_blur + 1)]
    parts.append("[src_bg]split=%d%s" % (n_blur + 1, "".join(f"[{t}]" for t in tags)))
    for k in range(1, n_blur + 1):
        parts.append(f"[b{k}]gblur=sigma={_fmt(sigmas[k - 1])}[bg{k}]")

    parts.append(f"[3:v]format=gray,scale={w}:{h}:flags=bilinear[dep]")
    for k in range(1, n_blur + 1):
        t = step * k - soft / 2.0
        parts.append(
            f"[dep]lut=y='clip((val-{_fmt(t)})/{_fmt(soft)}*255,0,255)',format=gray[c{k}]"
        )

    acc = "b0"
    for k in range(1, n_blur + 1):
        nxt = f"acc{k}"
        parts.append(f"[{acc}][bg{k}][c{k}]maskedmerge=planes=7[{nxt}]")
        acc = nxt
    return parts, acc


def build_filtergraph(
    src: VideoInfo,
    mask_w: int,
    mask_h: int,
    p: EffectParams,
    has_depth: bool = False,
) -> str:
    """拼出 filter_complex。

    输入：``0``=原视频，``1``=人物掩码，``2``=存在门，``3``=深度（可选）。
    给了 ``has_depth`` 时，背景走「分层景深虚化」；否则维持原来的全局均匀虚化。
    """
    w, h = src.width, src.height

    mask_chain = [f"scale={w}:{h}:flags=bicubic", "format=gray"]
    if p.temporal_smooth:
        mask_chain.append("tmix=frames=3:weights='1 2 1'")
    if p.feather > 0:
        mask_chain.append(f"gblur=sigma={_fmt(p.feather)}")
    mask_chain.append("format=gray")
    mask = f"[1:v]" + ",".join(mask_chain) + "[pm]"

    gate = f"[2:v]scale={w}:{h}:flags=bilinear,format=gray,format=yuv420p[pg]"

    fg_chain = [
        f"eq=brightness={_fmt(p.fg_brightness)}:contrast={_fmt(p.fg_contrast)}"
        f":saturation={_fmt(p.fg_saturation)}"
    ]
    if p.fg_sharpen > 0:
        fg_chain.append(f"unsharp=luma_msize_x=5:luma_msize_y=5:luma_amount={_fmt(p.fg_sharpen)}")

    bg_eq = f"eq=brightness={_fmt(p.bg_brightness)}:saturation={_fmt(p.bg_saturation)}"

    parts: list[str] = []
    parts.append("[0:v]format=yuv420p,split=3[src_bg][src_fg][src_gate]")

    # 背景：有深度就分层虚化，否则保持原来的全局均匀虚化
    if has_depth:
        chain, acc = _depth_blur_chain(p, w, h)
        parts.extend(chain)
        parts.append(f"[{acc}]{bg_eq}[bg_blur]")
    else:
        parts.append(f"[src_bg]gblur=sigma={_fmt(p.blur)},{bg_eq}[bg_blur]")

    # 存在门：maskedmerge 的掩码为 255 时取第二路输入，所以
    # 「目标在画面里（gate=255）→ 取虚化背景；不在（gate=0）→ 取原画」。
    parts.append("[src_gate][bg_blur][pg]maskedmerge=planes=7[bg_gated]")
    # 主体：提亮 + 可选柔光
    parts.append(f"[src_fg]{','.join(fg_chain)}[fg_pre]")
    if p.glow_opacity > 0:
        parts.append("[fg_pre]split=2[fg_a][fg_b]")
        parts.append(
            f"[fg_b]gblur=sigma={_fmt(p.glow_sigma)},eq=brightness=0.12[fg_glow]"
        )
        parts.append(
            f"[fg_a][fg_glow]blend=all_mode=screen:all_opacity={_fmt(p.glow_opacity)}[fg]"
        )
    else:
        parts.append("[fg_pre]null[fg]")
    parts.append(mask)
    parts.append(gate)
    parts.append("[bg_gated][fg][pm]maskedmerge=planes=7[vout]")
    return ";".join(parts)


def render(
    src_video: str | Path,
    mask_video: str | Path,
    gate_video: str | Path,
    out_path: str | Path,
    info: VideoInfo,
    mask_size: tuple[int, int],
    params: EffectParams,
    encoder: str = "auto",
    quality: int = 60,
    audio: bool = True,
    depth_video: str | Path | None = None,
) -> None:
    """执行合成 + 编码。

    ``depth_video`` 给了就走「真景深」：背景按深度分层虚化，近处轻、远处重。
    """
    mw, mh = mask_size
    graph = build_filtergraph(info, mw, mh, params, has_depth=depth_video is not None)

    if encoder == "auto":
        encoder = "h264_videotoolbox"

    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-i", str(src_video),
        "-i", str(mask_video),
        "-i", str(gate_video),
    ]
    if depth_video is not None:
        cmd += ["-i", str(depth_video)]
    cmd += ["-filter_complex", graph, "-map", "[vout]"]
    if info.has_audio and audio:
        cmd += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-an"]

    if encoder == "h264_videotoolbox":
        # videotoolbox 的 -q:v 是 0-100，越大质量越高
        cmd += ["-c:v", "h264_videotoolbox", "-q:v", str(int(max(1, min(100, quality)))), "-allow_sw", "1"]
    elif encoder == "hevc_videotoolbox":
        cmd += [
            "-c:v", "hevc_videotoolbox", "-q:v", str(int(max(1, min(100, quality)))),
            "-tag:v", "hvc1", "-allow_sw", "1",
        ]
    else:
        cmd += ["-c:v", encoder, "-crf", "18", "-preset", "veryfast"]

    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(out_path)]

    proc = _run(cmd)
    if proc.returncode != 0:
        raise VideoError(
            "合成失败：" + proc.stderr.decode(errors="replace")[:900] + f"\nfiltergraph: {graph}"
        )
    if not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
        raise VideoError("合成失败：输出文件为空")


def reencode_pair(
    video_a: str | Path,
    audio_source: str | Path,
    out_path: str | Path,
    info: VideoInfo,
    encoder: str = "auto",
    quality: int = 60,
    audio: bool = True,
) -> None:
    """把已处理好的画面（通常无音轨）与原片音频合成为最终成片。

    清除旁人模式下，``cleaned.mkv`` 已经是「保留目标 + 抹掉旁人并补回背景」的成品，
    整段都应保持清晰——**不做**虚化/压暗/去色，也**不受**存在门影响（存在门在目标
    离场时会切回原片，反而会让被清除的人重新冒出来）。这里只把无损中间产物重新编码，
    并合入原片音轨。
    """
    if encoder == "auto":
        encoder = "h264_videotoolbox"

    cmd = [FFMPEG, "-v", "error", "-nostdin", "-i", str(video_a)]
    if info.has_audio and audio:
        # 第二路输入取原片音轨；第一路只取视频。
        cmd += ["-i", str(audio_source), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
    else:
        cmd += ["-map", "0:v:0", "-an"]

    if encoder == "h264_videotoolbox":
        cmd += ["-c:v", "h264_videotoolbox", "-q:v", str(int(max(1, min(100, quality)))), "-allow_sw", "1"]
    elif encoder == "hevc_videotoolbox":
        cmd += [
            "-c:v", "hevc_videotoolbox", "-q:v", str(int(max(1, min(100, quality)))),
            "-tag:v", "hvc1", "-allow_sw", "1",
        ]
    else:
        cmd += ["-c:v", encoder, "-crf", "18", "-preset", "veryfast"]

    cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(out_path)]

    proc = _run(cmd)
    if proc.returncode != 0:
        raise VideoError("合成失败：" + proc.stderr.decode(errors="replace")[:900])
    if not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
        raise VideoError("合成失败：输出文件为空")


def build_preview_graph(src: VideoInfo, mask_size: tuple[int, int], p: EffectParams) -> str:
    """单帧预览用的 filtergraph（只处理一帧，不需要存在门）。"""
    w, h = src.width, src.height
    chain = [f"scale={w}:{h}:flags=bicubic", "format=gray"]
    if p.feather > 0:
        chain.append(f"gblur=sigma={_fmt(p.feather)}")
    chain.append("format=gray")
    return ";".join([
        "[0:v]format=yuv420p,split=2[src_bg][src_fg]",
        f"[src_bg]gblur=sigma={_fmt(p.blur)},"
        f"eq=brightness={_fmt(p.bg_brightness)}:saturation={_fmt(p.bg_saturation)}[bg]",
        f"[src_fg]eq=brightness={_fmt(p.fg_brightness)}:contrast={_fmt(p.fg_contrast)}"
        f":saturation={_fmt(p.fg_saturation)}[fg]",
        f"[1:v]{','.join(chain)}[pm]",
        "[bg][fg][pm]maskedmerge=planes=7[vout]",
    ])
