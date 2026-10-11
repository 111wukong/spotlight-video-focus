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


def build_filtergraph(src: VideoInfo, mask_w: int, mask_h: int, p: EffectParams) -> str:
    """拼出 filter_complex。输入 0=原视频，1=人物掩码，2=存在门。"""
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

    parts: list[str] = []
    parts.append(f"[0:v]format=yuv420p,split=3[src_bg][src_fg][src_gate]")
    # 背景：虚化 + 轻微压暗去色
    parts.append(
        f"[src_bg]gblur=sigma={_fmt(p.blur)},"
        f"eq=brightness={_fmt(p.bg_brightness)}:saturation={_fmt(p.bg_saturation)}[bg_blur]"
    )
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
) -> None:
    """执行合成 + 编码。"""
    mw, mh = mask_size
    graph = build_filtergraph(info, mw, mh, params)

    if encoder == "auto":
        encoder = "h264_videotoolbox"

    cmd = [
        FFMPEG, "-v", "error", "-nostdin",
        "-i", str(src_video),
        "-i", str(mask_video),
        "-i", str(gate_video),
        "-filter_complex", graph,
        "-map", "[vout]",
    ]
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
