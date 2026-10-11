"""Spotlight 命令行入口。

::

    python -m spotlight.cli doctor                          # 环境自检
    python -m spotlight.cli download --preset balanced      # 下载权重
    python -m spotlight.cli preview -i in.mp4 --frame 30 --point 0.4,0.3
    python -m spotlight.cli run -i in.mp4 -o out.mp4 --frame 30 --box 0.17,0.08,0.66,1.0
    python -m spotlight.cli serve                           # 启动网页界面
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS = REPO_ROOT / "models"


# --------------------------------------------------------------------- 工具


def _parse_points(values: list[str]) -> tuple[list[tuple[float, float]], list[int]]:
    points: list[tuple[float, float]] = []
    labels: list[int] = []
    for raw in values:
        sign = 1
        text = raw
        if text.startswith("-"):
            sign, text = 0, text[1:]
        elif text.startswith("+"):
            text = text[1:]
        parts = text.split(",")
        if len(parts) != 2:
            raise SystemExit(f"坐标格式应为 x,y，收到 {raw!r}")
        points.append((float(parts[0]), float(parts[1])))
        labels.append(sign)
    return points, labels


def _parse_box(value: str | None) -> tuple[float, float, float, float] | None:
    if not value:
        return None
    parts = [float(v) for v in value.split(",")]
    if len(parts) != 4:
        raise SystemExit("框的格式应为 x1,y1,x2,y2")
    return parts[0], parts[1], parts[2], parts[3]


def _to_norm(vals, w: int, h: int, mode: str):
    if mode == "px":
        return [(x / w, y / h) for x, y in vals]
    return vals


def _effects_from_args(a) -> "EffectParams":
    from .composite import EffectParams

    p = EffectParams()
    if a.blur is not None:
        p.blur = a.blur
    if a.brighten is not None:
        p.fg_brightness = a.brighten
    if a.contrast is not None:
        p.fg_contrast = a.contrast
    if a.glow is not None:
        p.glow_opacity = a.glow
    if a.feather is not None:
        p.feather = a.feather
    if a.bg_dim is not None:
        p.bg_brightness = a.bg_dim
    if a.bg_desat is not None:
        p.bg_saturation = 1.0 - a.bg_desat
    if a.sharpen is not None:
        p.fg_sharpen = a.sharpen
    if a.no_temporal_smooth:
        p.temporal_smooth = False
    return p


# --------------------------------------------------------------------- 子命令


def _parse_anchors(values: list[str] | None) -> list[tuple[int, list[tuple[float, float]], list[int], tuple | None]]:
    """解析 ``--anchor`` 参数。

    格式：``帧号:坐标``，坐标可以是点 ``x,y``（前缀 ``-`` 表示排除），
    也可以是框 ``box=x1,y1,x2,y2``。可以重复多次。
    例如：``--anchor 300:0.52,0.61``、``--anchor 480:box=0.3,0.2,0.6,0.9``
    """
    out = []
    for raw in values or []:
        if ":" not in raw:
            raise SystemExit(f"--anchor 格式应为 帧号:坐标，收到 {raw!r}")
        frame_text, spec = raw.split(":", 1)
        try:
            frame = int(frame_text)
        except ValueError:
            raise SystemExit(f"--anchor 的帧号不是整数：{frame_text!r}") from None
        spec = spec.strip()
        if spec.startswith("box="):
            box = _parse_box(spec[4:])
            out.append((frame, [], [], box))
        else:
            pts, labels = _parse_points([spec])
            out.append((frame, pts, labels, None))
    return out


def _print_estimate(n_frames: int, preset: str, frame_step: int, n_anchors: int) -> None:
    from .pipeline import estimate_seconds

    secs = estimate_seconds(n_frames, preset, frame_step)
    mins, s = divmod(int(secs), 60)
    human = f"{mins} 分 {s} 秒" if mins else f"{s} 秒"
    print(f"  预计耗时约 {human}（{n_frames} 帧 · {preset} 档 · 锚点 {n_anchors} 个）")


def cmd_doctor(a) -> int:
    print("Spotlight 环境自检")
    print("-" * 52)
    ok = True

    def line(name: str, good: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and good
        print(f"  {'✓' if good else '✗'} {name:<22} {detail}")

    import platform

    line("系统", platform.system() == "Darwin", f"{platform.system()} {platform.release()}")
    line("芯片架构", platform.machine() == "arm64", platform.machine())

    ffmpeg = shutil.which("ffmpeg")
    line("ffmpeg", bool(ffmpeg), ffmpeg or "未找到，请先 brew install ffmpeg")

    try:
        import mlx.core as mx

        line("MLX", True, f"{mx.__version__}，Metal {'可用' if mx.metal.is_available() else '不可用'}")
    except Exception as exc:
        line("MLX", False, f"导入失败：{exc}")

    try:
        import mlx_sam

        line("mlx-sam", True, "已安装")
    except Exception as exc:
        line("mlx-sam", False, f"导入失败：{exc}")

    try:
        from .appearance import _VISION_OK

        line("Vision 框架", bool(_VISION_OK), "可用（重识别已启用）" if _VISION_OK else "不可用（重识别降级）")
    except Exception as exc:
        line("Vision 框架", False, str(exc))

    from .cleanplate import resolve_lama
    from .inpaint_diffusion import resolve_diffusion
    lama = resolve_lama(a.models)
    line("LaMa 修复模型", lama is not None,
         f"{lama.stat().st_size/1e6:.0f} MB（清除旁人时用来补空洞）" if lama else
         "未下载（清除旁人功能需要：spotlight download --preset lama）")
    diff = resolve_diffusion(a.models)
    line("扩散补洞模型", diff is not None,
         "已就绪（清除旁人可选 --clean-inpaint-model diffusion）" if diff else
         "未下载（可选：spotlight download --preset diffusion，质量更高但慢）")

    from .segmenter import Segmenter, resolve_checkpoint

    for preset in Segmenter.PRESETS:
        try:
            path, image_size, long_side, _ = resolve_checkpoint(a.models, preset)
            size = Path(path).stat().st_size / 1e6
            line(f"权重 {preset}", True, f"{size:.0f} MB  {Path(path).name}")
        except FileNotFoundError:
            line(f"权重 {preset}", False, "未下载（spotlight download --preset " + preset + "）")

    print("-" * 52)
    print("环境就绪，可以开始。" if ok else "还有项目未就绪，请按上面的提示处理。")
    return 0 if ok else 1


def cmd_download(a) -> int:
    from .segmenter import Segmenter

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # WorkBuddy 会注入内部 lowercase proxy，但它不一定能访问 HuggingFace。
    # 本机 Clash (7897/7890) 可用时优先走它，并同步设置大小写，避免子进程选错代理。
    for cand in ("127.0.0.1:7897", "127.0.0.1:7890"):
        if _proxy_listening(cand):
            proxy = f"http://{cand}"
            for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                os.environ[key] = proxy
            print(f"  · 使用本机 Clash 代理 {cand}")
            break
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    if a.mirror:
        endpoint = a.mirror
        print(f"  · 使用镜像 {endpoint}")

    models_dir = Path(a.models)
    models_dir.mkdir(parents=True, exist_ok=True)
    rc = 0

    if a.preset == "lama":
        return _download_lama(models_dir, a.force)

    if a.preset == "diffusion":
        return _download_diffusion(models_dir, a.force)

    presets = list(Segmenter.PRESETS) if a.preset == "all" else [a.preset]
    for preset in presets:
        spec = Segmenter.PRESETS[preset]
        dest = models_dir / spec["subdir"]
        target = dest / spec["filename"]
        if target.is_file() and not a.force:
            print(f"  · {preset} 已存在，跳过（{target}）")
            continue
        url = f"{endpoint}/{spec['repo']}/resolve/main/{spec['filename']}"
        print(f"  · 下载 {preset} ← {spec['repo']}/{spec['filename']}")
        if _curl_download_file(url, target, token):
            print(f"    ✓ {target}")
        else:
            rc = 1
            print("    ✗ 失败（检查网络/代理后重试）")
    if a.preset == "all":
        rc = max(rc, _download_lama(models_dir, a.force))
        rc = max(rc, _download_diffusion(models_dir, a.force))
    return rc


def _proxy_listening(host_port: str) -> bool:
    import socket

    try:
        h, p = host_port.split(":")
        s = socket.create_connection((h, int(p)), timeout=1.5)
        s.close()
        return True
    except Exception:
        return False


def _curl_proxy_args() -> list[str]:
    """显式指定代理，避免 curl 优先拾取 WorkBuddy 注入的内部 lowercase proxy。"""
    proxy = (
        os.environ.get("SPOTLIGHT_HF_PROXY") or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
        or os.environ.get("https_proxy") or os.environ.get("http_proxy")
        or os.environ.get("ALL_PROXY") or os.environ.get("all_proxy")
    )
    return ["--proxy", proxy] if proxy else []


def _curl_download_file(url: str, local, token: str | None,
                        expected_size: int | None = None) -> bool:
    """用 curl 下载单文件，支持断点续传与自动重试（对代理抖动更稳）。"""
    from pathlib import Path as _P

    local = _P(local)
    local.parent.mkdir(parents=True, exist_ok=True)
    if expected_size is not None and local.is_file() and local.stat().st_size == expected_size:
        return True
    for _ in range(60):
        cmd = ["curl", *(_curl_proxy_args()), "-sS", "-L", "--location-trusted", "-C", "-", "-f",
               "-o", str(local), "--connect-timeout", "20", "--max-time", "1800",
               "--retry", "20", "--retry-all-errors", "--retry-delay", "3", "--retry-max-time", "600"]
        if token:
            cmd += ["-H", f"Authorization: Bearer {token}"]
        cmd.append(url)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=2000)
        except Exception:
            r = None
        if local.exists():
            sz = local.stat().st_size
            if expected_size is not None and sz == expected_size:
                return True
            if expected_size is None and sz > 0 and r is not None and r.returncode == 0:
                return True
    return False


def _curl_download_repo(repo: str, dest, endpoint: str, token: str | None,
                        force: bool = False) -> bool:
    from pathlib import Path as _P

    dest = _P(dest)
    api = f"{endpoint.rstrip('/')}/api/models/{repo}/tree/main?recursive=true"
    hdr = ["-H", f"Authorization: Bearer {token}"] if token else []
    try:
        r = subprocess.run(
            ["curl", *(_curl_proxy_args()), "-sS", "-L", "--location-trusted", "-f",
             "--connect-timeout", "20", "--max-time", "45", "--retry", "12",
             "--retry-all-errors", "--retry-delay", "2", "--retry-max-time", "120", *hdr, api],
            capture_output=True, text=True, timeout=180,
        )
    except Exception as exc:
        print(f"    ✗ 无法列举仓库文件（{repo}）：{type(exc).__name__}: {exc}")
        return False
    if r.returncode != 0 or not r.stdout.strip():
        print(f"    ✗ 无法列举仓库文件（{repo}）：{r.stderr[:200]}")
        return False
    try:
        tree = json.loads(r.stdout)
    except Exception as exc:
        print(f"    ✗ 仓库文件清单解析失败：{exc}")
        return False
    files = [e for e in tree if e.get("type") == "file"]
    total = sum(e.get("size", 0) for e in files)
    print(f"    共 {len(files)} 个文件，约 {total / 1024 / 1024:.0f} MB")
    ok = True
    for e in files:
        path = e["path"]
        size = e.get("size")
        if size is None:
            print(f"    ✗ 清单缺少文件大小：{path}")
            ok = False
            continue
        local = dest / path
        if local.is_file() and local.stat().st_size == size and not force:
            continue
        if force and local.is_file():
            local.unlink()
        url = f"{endpoint.rstrip('/')}/{repo}/resolve/main/{path}"
        print(f"    ↓ {path}  ({size / 1024 / 1024:.1f} MB)", flush=True)
        if not _curl_download_file(url, dest / path, token, expected_size=size):
            ok = False
            print(f"    ✗ 失败 {path}")
    return ok


def _download_lama(models_dir: Path, force: bool) -> int:
    """下载「清除旁人」用的 LaMa 修复模型。"""
    from .cleanplate import resolve_lama

    if resolve_lama(models_dir) is not None and not force:
        print(f"  · LaMa 已存在，跳过（{resolve_lama(models_dir)}）")
        return 0
    dest = models_dir / "lama"
    dest.mkdir(parents=True, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    print("  · 下载 LaMa 修复模型 ← Carve/LaMa-ONNX/lama_fp32.onnx（约 208 MB）")
    if _curl_download_file(f"{endpoint}/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
                           dest / "lama_fp32.onnx", token):
        print(f"    ✓ {dest / 'lama_fp32.onnx'}")
        return 0
    print("    ✗ 失败（检查网络/代理后重试）")
    return 1


def _download_diffusion(models_dir: Path, force: bool) -> int:
    """下载「清除旁人」用的扩散补洞模型（SD inpainting / diffusers 格式，约 2.6 GB）。"""
    from .inpaint_diffusion import resolve_diffusion

    if resolve_diffusion(models_dir) is not None and not force:
        print(f"  · 扩散补洞模型已存在，跳过（{resolve_diffusion(models_dir)}）")
        return 0
    dest = models_dir / "diffusion"
    dest.mkdir(parents=True, exist_ok=True)
    # 原 stabilityai/stable-diffusion-2-inpainting 已从 HuggingFace 下架（404），
    # 改用同样为 diffusers InpaintPipeline 格式、写实风格的 Sanster 替代品。
    repo = "Sanster/Realistic_Vision_V1.4-inpainting"
    # 替代模型公开可下载，无需把个人 HF token 转发到外部 CDN。
    endpoint = os.environ.get("HF_ENDPOINT") or "https://huggingface.co"
    print(f"  · 下载扩散补洞模型 ← {repo}（约 2.6 GB，SD inpainting / diffusers）")
    if _curl_download_repo(repo, dest, endpoint, token=None, force=force):
        print(f"    ✓ {dest}")
        return 0
    print("    ✗ 失败（检查网络/代理后重试）；装好模型后清除旁人加 --clean-inpaint-model diffusion 即可启用。")
    return 1


def cmd_preview(a) -> int:
    from .pipeline import Selection, preview_mask
    from .video import probe

    points, labels = _parse_points(a.point or [])
    box = _parse_box(a.box)
    if not points and box is None:
        raise SystemExit("请用 --point x,y 或 --box x1,y1,x2,y2 指定目标")

    info = probe(a.input)
    if a.coords == "px":
        points = _to_norm(points, info.width, info.height, "px")
        box = _to_norm([box[:2], box[2:]], info.width, info.height, "px") if box else None
        if box is not None:
            box = (box[0][0], box[0][1], box[1][0], box[1][1])

    soft, size, _, note = preview_mask(a.input, Selection(a.frame, points, labels, box), a.models, a.preset, a.work_long_side)
    if note:
        print(note)
    out = Path(a.output)
    import cv2
    import numpy as np

    overlay = None
    if a.overlay:
        from .video import FrameStore
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            store = FrameStore(a.input, td, long_side=a.work_long_side, info=info)
            rgb = np.asarray(store.frame(a.frame)).copy()
            m = (soft.astype(np.float32) / 255.0)[..., None]
            tint = np.zeros_like(rgb)
            tint[..., 0] = 255
            overlay = (rgb * (1 - 0.45 * m) + tint * (0.45 * m)).astype(np.uint8)
            store.close()
        cv2.imwrite(str(out), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(out), soft)
    area = int((soft > 100).sum())
    print(f"掩码已写出 {out}（工作分辨率 {size[0]}x{size[1]}，覆盖 {area} 像素）")
    return 0


def cmd_run(a) -> int:
    from .pipeline import RunConfig, Selection, run
    from .tracker import GuardConfig
    from .video import probe

    points, labels = _parse_points(a.point or [])
    box = _parse_box(a.box)
    if not points and box is None:
        raise SystemExit("请用 --point x,y 或 --box x1,y1,x2,y2 指定目标")

    info = probe(a.input)
    if a.coords == "px":
        points = _to_norm(points, info.width, info.height, "px")
        if box is not None:
            nb = _to_norm([box[:2], box[2:]], info.width, info.height, "px")
            box = (nb[0][0], nb[0][1], nb[1][0], nb[1][1])

    selections = [Selection(a.frame, points, labels, box)]
    for frame, pts, labs, bx in _parse_anchors(a.anchor):
        if a.coords == "px":
            pts = _to_norm(pts, info.width, info.height, "px")
            if bx is not None:
                nb = _to_norm([bx[:2], bx[2:]], info.width, info.height, "px")
                bx = (nb[0][0], nb[0][1], nb[1][0], nb[1][1])
        selections.append(Selection(frame, pts, labs, bx))

    frame_start = frame_end = None
    if a.range:
        try:
            s, e = a.range.split(":")
            frame_start = int(s) if s else None
            frame_end = int(e) if e else None
        except ValueError:
            raise SystemExit("--range 格式应为 起始帧:结束帧，例如 0:900") from None

    cfg = RunConfig(
        preset=a.preset,
        effects=_effects_from_args(a),
        guard=GuardConfig(),
        encoder=a.encoder,
        quality=a.quality,
        work_long_side=a.work_long_side,
        frame_step=a.frame_step,
        max_frames=a.max_frames,
        frame_start=frame_start,
        frame_end=frame_end,
        audio=not a.no_audio,
        reid=not a.no_reid,
        remove_others=bool(a.remove_people),
    )
    if a.remove_people:
        from .cleanplate import CleanConfig
        cfg.clean = CleanConfig(segment=a.clean_segment, window=a.clean_window,
                                dilate=a.clean_dilate, inpaint=not a.no_inpaint,
                                inpaint_model=a.clean_inpaint_model)
    if a.no_reid:
        cfg.guard.identity_floor = -1.0
        cfg.guard.reanchor_score = 2.0

    span = (frame_end or info.n_frames) - (frame_start or 0)
    if not a.quiet:
        _print_estimate(span, a.preset, a.frame_step, len(selections))

    last = {"stage": None}

    def on_progress(p):
        if a.quiet and p.stage != "render":
            return
        if p.stage == last["stage"] and p.done % 10:
            return
        last["stage"] = p.stage
        bar = ""
        if p.total:
            filled = int(28 * p.done / p.total)
            bar = "[" + "#" * filled + "." * (28 - filled) + "] "
        extra = ""
        if p.extra.get("reanchors"):
            extra = f"  重锚定 {p.extra['reanchors']} 次"
        print(f"  {bar}{p.message}{extra}", flush=True)

    res = run(a.input, selections, a.output, a.models, cfg, on_progress=on_progress)
    print()
    print(f"✓ 输出 {res.output_path}")
    print(f"  {res.n_frames} 帧 / {res.elapsed:.1f} 秒 → {res.frames_per_second:.1f} 帧/秒")
    g = res.guard_summary
    if g["lost_frames"] or g.get("suspect_frames"):
        print(f"  可疑 {g.get('suspect_frames', 0)} 帧，确认丢失 {g['lost_frames']} 帧，重锚定 {g['reanchors']} 次")
    if res.lost_segments:
        spans = "、".join(f"{s['start']}~{s['end']}" for s in res.lost_segments[:5])
        more = f" 等 {len(res.lost_segments)} 段" if len(res.lost_segments) > 5 else ""
        print(f"  目标不在画面里的片段：{spans}{more}")
        print("  这些片段原样输出，没有做虚化。跑偏的话可以用 --anchor 帧号:x,y 补一个修正点。")
    return 0


def cmd_serve(a) -> int:
    from .server import serve

    serve(host=a.host, port=a.port, models_dir=a.models, open_browser=not a.no_browser)
    return 0


# --------------------------------------------------------------------- 解析


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spotlight",
        description="视频单人聚焦：只突出你选中的那个人，其余全部虚化。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("-i", "--input", required=False, help="输入视频")
        sp.add_argument("--models", default=str(DEFAULT_MODELS), help="权重目录")
        sp.add_argument("--preset", default="balanced", choices=["turbo", "fast", "balanced", "quality"], help="处理档位")
        sp.add_argument("--work-long-side", type=int, default=None, help="工作分辨率长边（默认随档位）")

    d = sub.add_parser("doctor", help="环境自检")
    d.add_argument("--models", default=str(DEFAULT_MODELS))
    d.set_defaults(func=cmd_doctor)

    dl = sub.add_parser("download", help="下载模型权重")
    dl.add_argument("--preset", default="balanced",
                    choices=["turbo", "fast", "balanced", "quality", "lama", "diffusion", "all"],
                    help="lama = 「清除旁人」LaMa 修复模型；diffusion = 扩散补洞模型(SD inpainting，数 GB)；all = 全部")
    dl.add_argument("--models", default=str(DEFAULT_MODELS))
    dl.add_argument("--mirror", default="", help="HuggingFace 镜像（默认直连 huggingface.co）；例如 https://hf-mirror.com")
    dl.add_argument("--force", action="store_true")
    dl.set_defaults(func=cmd_download)

    pv = sub.add_parser("preview", help="只处理一帧，检查选人是否正确")
    add_common(pv)
    pv.add_argument("--frame", type=int, default=0)
    pv.add_argument("--point", action="append", help="目标点 x,y；前缀 - 表示负样本")
    pv.add_argument("--box", help="目标框 x1,y1,x2,y2")
    pv.add_argument("--coords", default="norm", choices=["norm", "px"])
    pv.add_argument("-o", "--output", default="preview_mask.png")
    pv.add_argument("--overlay", action="store_true", help="叠加到原图上更直观")
    pv.set_defaults(func=cmd_preview)

    r = sub.add_parser("run", help="处理整段视频")
    add_common(r)
    r.add_argument("-o", "--output", required=True, help="输出视频")
    r.add_argument("--frame", type=int, default=0, help="在哪一帧上选人")
    r.add_argument("--point", action="append", help="目标点 x,y；前缀 - 表示负样本")
    r.add_argument("--box", help="目标框 x1,y1,x2,y2")
    r.add_argument("--coords", default="norm", choices=["norm", "px"])
    r.add_argument("--blur", type=float, default=None, help="背景虚化强度（默认 26）")
    r.add_argument("--brighten", type=float, default=None, help="主体提亮（默认 0.055）")
    r.add_argument("--contrast", type=float, default=None, help="主体对比度（默认 1.07）")
    r.add_argument("--glow", type=float, default=None, help="柔光强度 0-1（默认 0.22）")
    r.add_argument("--feather", type=float, default=None, help="边缘羽化（默认 3）")
    r.add_argument("--bg-dim", type=float, default=None, help="背景压暗，如 -0.08")
    r.add_argument("--bg-desat", type=float, default=None, help="背景去色 0-1，如 0.3")
    r.add_argument("--sharpen", type=float, default=None, help="主体锐化，如 0.8")
    r.add_argument("--no-temporal-smooth", action="store_true", help="关闭掩码时域平滑")
    r.add_argument("--frame-step", type=int, default=1, help="跳帧推理（>1 更快但更抖）")
    r.add_argument("--max-frames", type=int, default=None)
    r.add_argument("--range", dest="range", default=None, help="只处理某一段：起始帧:结束帧，如 0:900")
    r.add_argument(
        "--anchor", action="append", default=None,
        help="在别的帧上补修正点（可重复）：帧号:x,y（前缀 - 表示排除）或 帧号:box=x1,y1,x2,y2",
    )
    r.add_argument("--encoder", default="auto", help="auto / h264_videotoolbox / hevc_videotoolbox / libx264")
    r.add_argument("--quality", type=int, default=60, help="videotoolbox 质量 1-100")
    r.add_argument("--no-audio", action="store_true")
    r.add_argument("--no-reid", action="store_true", help="关闭外观重识别守卫")
    r.add_argument("--remove-people", action="store_true",
                   help="把除目标外的所有人抹掉，并用重建的背景补上（清除旁人）")
    r.add_argument("--clean-segment", type=int, default=32, help="每多少帧换一张背景板（默认 32）")
    r.add_argument("--clean-window", type=int, default=16,
                   help="背景板取样窗口（中心帧前后各多少帧，默认 16）。"
                        "调大能少一些空洞，但视差重影会明显变重")
    r.add_argument("--clean-dilate", type=int, default=7, help="移除掩码膨胀像素（默认 7）")
    r.add_argument("--no-inpaint", action="store_true",
                   help="清除旁人时不用 LaMa 补空洞（空洞会留黑，一般不用关）")
    r.add_argument("--clean-inpaint-model", default="lama", choices=["lama", "diffusion"],
                   help="补洞模型：lama（默认·快）或 diffusion（SD inpainting·质量更高但慢，"
                        "需先 spotlight download --preset diffusion）")
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("serve", help="启动本地网页界面")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8760)
    s.add_argument("--models", default=str(DEFAULT_MODELS))
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "func", None) is cmd_doctor:
        return args.func(args)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
