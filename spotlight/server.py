"""本地网页界面后端。

流程：上传视频 → 在画面上点选/框选目标 → 立即看到抠像预览 → 开始处理 →
实时进度 → 在线预览与下载成片。

所有数据都留在本机，服务只监听 ``127.0.0.1``。
"""

from __future__ import annotations

import base64
import json
import queue
import shutil
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from .appearance import _VISION_OK
from .cleanplate import CleanConfig, resolve_lama
from .inpaint_diffusion import resolve_diffusion
from .depth import resolve_depth
from .composite import EffectParams
from .pipeline import (
    Cancelled,
    Progress,
    RunConfig,
    Selection,
    PRESET_SPEED,
    estimate_seconds,
    preview_mask,
    run,
)
from .segmenter import Segmenter, get_segmenter
from .tracker import GuardConfig
from .video import FrameStore, VideoError, extract_jpeg, probe

WEB_DIR = Path(__file__).resolve().parent / "web"
WORK_ROOT = Path(__file__).resolve().parent.parent / "work"
MAX_UPLOAD_BYTES = 4 * 1024**3


@dataclass
class VideoEntry:
    vid: str
    path: Path
    dir: Path
    width: int
    height: int
    fps: float
    fps_expr: str
    n_frames: int
    duration: float
    has_audio: bool
    store: FrameStore | None = None
    store_key: int | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class Job:
    jid: str
    vid: str
    status: str = "queued"          # queued | running | done | error | cancelled
    progress: Progress = field(default_factory=lambda: Progress("queued"))
    output: Path | None = None
    error: str = ""
    result: dict = field(default_factory=dict)
    cancel: threading.Event = field(default_factory=threading.Event)
    started: float = 0.0
    finished: float = 0.0


class Hub:
    def __init__(self, models_dir: Path) -> None:
        self.models_dir = models_dir
        self.videos: dict[str, VideoEntry] = {}
        self.jobs: dict[str, Job] = {}
        self.subscribers: dict[str, list[queue.Queue]] = {}
        #: 预览与正式处理会共用同一个（带缓存的）Segmenter 实例，
        #: MLX 模型对象不适合并发调用，用一把锁串起来；预览用非阻塞获取，
        #: 处理中就直接告诉前端「忙」，而不是悄悄跑出错误结果。
        self.infer_lock = threading.Lock()
        self._lock = threading.Lock()

    # ------------------------------------------------------------ 视频

    def add_video(self, path: Path, vid: str | None = None) -> VideoEntry:
        info = probe(path)
        vid = vid or uuid.uuid4().hex[:12]
        entry = VideoEntry(
            vid=vid, path=path, dir=path.parent,
            width=info.width, height=info.height, fps=info.fps, fps_expr=info.fps_expr,
            n_frames=info.n_frames, duration=info.duration, has_audio=info.has_audio,
        )
        with self._lock:
            self.videos[vid] = entry
        return entry

    def get_video(self, vid: str) -> VideoEntry:
        entry = self.videos.get(vid)
        if entry is None:
            raise HTTPException(404, "视频不存在，请重新上传")
        return entry

    def store_for(self, entry: VideoEntry, preset: str, work_long_side: int | None = None) -> FrameStore:
        """按需建立（并缓存）帧存储。缓存的是磁盘 memmap，占用很小。

        缓存键带上工作分辨率：不同档位解码出的帧尺寸不同，混用会让
        预览和处理结果的掩码对不上。
        """
        _, long_side = get_segmenter(self.models_dir, preset, work_long_side)
        with entry.lock:
            if entry.store is None or entry.store_key != long_side:
                if entry.store is not None:
                    entry.store.close()
                entry.store = FrameStore(entry.path, entry.dir / "frames", long_side=long_side)
                entry.store_key = long_side
            return entry.store

    # ------------------------------------------------------------ 任务

    def create_job(self, vid: str) -> Job:
        job = Job(jid=uuid.uuid4().hex[:12], vid=vid)
        with self._lock:
            self.jobs[job.jid] = job
            self.subscribers[job.jid] = []
        return job

    def publish(self, job: Job, prog: Progress) -> None:
        job.progress = prog
        payload = {
            "stage": prog.stage, "done": prog.done, "total": prog.total,
            "fraction": prog.fraction, "message": prog.message, **prog.extra,
        }
        for q in list(self.subscribers.get(job.jid, [])):
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass

    def finish(self, job: Job, payload: dict) -> None:
        for q in list(self.subscribers.get(job.jid, [])):
            try:
                q.put_nowait(payload)
            except queue.Full:
                pass


def _effects_from_payload(data: dict) -> EffectParams:
    p = EffectParams()
    mapping = {
        "blur": "blur", "brighten": "fg_brightness", "contrast": "fg_contrast",
        "saturation": "fg_saturation", "glow": "glow_opacity", "glow_sigma": "glow_sigma",
        "feather": "feather", "bg_dim": "bg_brightness", "bg_desat": "bg_saturation",
        "sharpen": "fg_sharpen",
    }
    for key, attr in mapping.items():
        if key not in data or data[key] is None:
            continue
        value = float(data[key])
        if key == "bg_desat":
            value = 1.0 - value  # 前端给的是「去色程度」，ffmpeg 要的是饱和度
        setattr(p, attr, value)
    if "temporal_smooth" in data:
        p.temporal_smooth = bool(data["temporal_smooth"])
    if "fill_holes" in data:
        p.fill_holes = bool(data["fill_holes"])
    return p


def _selection_from_payload(data: dict) -> Selection:
    points = [(float(p[0]), float(p[1])) for p in (data.get("points") or [])]
    labels = [int(v) for v in (data.get("labels") or [])]
    box = data.get("box")
    # 前端锚点对象用 ``frame``，单点接口用 ``frame_idx``，两个都认
    frame = data.get("frame_idx")
    if frame is None:
        frame = data.get("frame")
    return Selection(
        frame_idx=int(frame or 0),
        points=points,
        labels=labels or [1] * len(points),
        box=tuple(float(v) for v in box) if box else None,
    )


def _selections_from_payload(data: dict) -> list[Selection]:
    """从请求里取出全部锚点：``anchors`` 优先，否则退回单个 ``frame_idx`` 形式。"""
    raw = data.get("anchors")
    out: list[Selection] = []
    if isinstance(raw, list) and raw:
        for item in raw:
            if not isinstance(item, dict):
                continue
            try:
                out.append(_selection_from_payload(item))
            except (TypeError, ValueError):
                continue
    if not out:
        out.append(_selection_from_payload(data))
    return out


def _overlay_png(rgb: np.ndarray, soft: np.ndarray, color=(0, 200, 255), alpha: float = 0.42) -> str:
    m = (soft.astype(np.float32) / 255.0)[..., None]
    tint = np.zeros_like(rgb, dtype=np.float32)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    out = rgb.astype(np.float32) * (1 - alpha * m) + tint * (alpha * m)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(500, "生成预览失败")
    return base64.b64encode(buf.tobytes()).decode()


def _prune_workspace(root: Path, keep: int = 12, max_age_days: float = 7.0) -> int:
    """清理旧的上传目录。

    每上传一个视频就会在 ``work/uploads/<vid>`` 留下原片、解码帧、缩略图和成片，
    不清理的话会一直堆积。这里按「最多保留最近 keep 个」+「超过 max_age_days 天」
    两条规则删除，只在服务启动时跑一次，不会打断正在进行的任务。
    """
    uploads = root / "uploads"
    if not uploads.is_dir():
        return 0
    entries = []
    now = time.time()
    for child in uploads.iterdir():
        if not child.is_dir():
            continue
        try:
            entries.append((child.stat().st_mtime, child))
        except OSError:
            continue
    entries.sort(reverse=True)
    removed = 0
    for idx, (mtime, child) in enumerate(entries):
        too_many = idx >= keep
        too_old = (now - mtime) > max_age_days * 86400
        if too_many or too_old:
            shutil.rmtree(child, ignore_errors=True)
            removed += 1
    return removed


def create_app(models_dir: Path) -> FastAPI:
    app = FastAPI(title="Spotlight", docs_url=None, redoc_url=None)
    hub = Hub(Path(models_dir))
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        pruned = _prune_workspace(WORK_ROOT)
        if pruned:
            print(f"已清理 {pruned} 个旧的上传目录")
    except Exception:  # pragma: no cover - 清理失败不应影响启动
        pass

    # ---------------------------------------------------------------- 页面

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        page = WEB_DIR / "index.html"
        if not page.is_file():
            raise HTTPException(500, "前端文件缺失")
        return HTMLResponse(page.read_text(encoding="utf-8"))

    @app.get("/api/status")
    def status() -> dict:
        presets = []
        for name, spec in Segmenter.PRESETS.items():
            path = Path(models_dir) / spec["subdir"] / spec["filename"]
            presets.append({
                "key": name, "label": spec["label"], "ready": path.is_file(),
                "image_size": spec["image_size"], "long_side": spec["long_side"],
                "size_mb": round(path.stat().st_size / 1e6) if path.is_file() else 0,
                "speed": PRESET_SPEED.get(name, 2.9),
            })
        return {
            "version": "0.1.0",
            "presets": presets,
            "vision": bool(_VISION_OK),
            "reid": bool(_VISION_OK),
            "clean": bool(_VISION_OK),
            "lama": resolve_lama(models_dir) is not None,
            "diffusion": resolve_diffusion(models_dir) is not None,
            "depth": resolve_depth(models_dir) is not None,
        }

    # ---------------------------------------------------------------- 上传

    @app.post("/api/upload")
    async def upload(file: UploadFile) -> dict:
        vid = uuid.uuid4().hex[:12]
        target_dir = WORK_ROOT / "uploads" / vid
        target_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(file.filename or "input.mp4").suffix or ".mp4"
        target = target_dir / f"input{suffix}"

        size = 0
        with open(target, "wb") as fh:
            while True:
                chunk = await file.read(1 << 22)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    fh.close()
                    shutil.rmtree(target_dir, ignore_errors=True)
                    raise HTTPException(413, "文件过大（上限 4 GB）")
                fh.write(chunk)

        try:
            entry = hub.add_video(target, vid=vid)
        except VideoError as exc:
            shutil.rmtree(target_dir, ignore_errors=True)
            raise HTTPException(400, str(exc)) from exc

        return {
            "vid": entry.vid,
            "width": entry.width, "height": entry.height,
            "fps": round(entry.fps, 4), "n_frames": entry.n_frames,
            "duration": round(entry.duration, 3), "has_audio": entry.has_audio,
            "size_mb": round(size / 1e6, 1),
        }

    @app.get("/api/video/{vid}")
    def video(vid: str, request: Request) -> FileResponse:
        entry = hub.get_video(vid)
        return FileResponse(entry.path, media_type="video/mp4")

    @app.get("/api/frame/{vid}/{idx}.jpg")
    def frame(vid: str, idx: int, w: int = 1280) -> FileResponse:
        entry = hub.get_video(vid)
        idx = max(0, min(max(0, entry.n_frames - 1), idx))
        out = entry.dir / "thumbs" / f"{idx}_{w}.jpg"
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            try:
                extract_jpeg(entry.path, idx, entry.fps, out, width=w)
            except VideoError as exc:
                raise HTTPException(500, str(exc)) from exc
        return FileResponse(out, media_type="image/jpeg")

    # ---------------------------------------------------------------- 选人

    @app.post("/api/select")
    def select(data: dict) -> JSONResponse:
        entry = hub.get_video(str(data.get("vid")))
        preset = str(data.get("preset") or "balanced")
        sel = _selection_from_payload(data)
        sel.frame_idx = max(0, min(max(0, entry.n_frames - 1), sel.frame_idx))

        store = hub.store_for(entry, preset, data.get("work_long_side"))
        if sel.frame_idx >= store.n_frames:
            raise HTTPException(400, "该帧超出可处理范围")

        if not hub.infer_lock.acquire(blocking=False):
            raise HTTPException(409, "正在处理另一个任务，等它跑完再选人")

        try:
            seg, _ = get_segmenter(models_dir, preset, data.get("work_long_side"))
            state = seg.new_state(store)
            from .appearance import AppearanceEngine
            from .pipeline import resolve_prompt
            from .segmenter import logits_to_soft

            logits, note = resolve_prompt(seg, state, AppearanceEngine(), store, sel)
            soft = logits_to_soft(logits)
            area = int((soft > 100).sum())
            ratio = area / float(store.width * store.height)
            rgb = np.asarray(store.frame(sel.frame_idx)).copy()
        finally:
            hub.infer_lock.release()

        hint = "" if area >= 64 else "没分割出有效目标，换个位置再点一次"
        return JSONResponse({
            "frame_idx": sel.frame_idx,
            "mask_png": _overlay_png(rgb, soft),
            "area": area,
            "area_ratio": round(ratio, 4),
            "ok": area >= 64,
            "hint": hint,
            "note": note,
        })

    # ---------------------------------------------------------------- 处理

    @app.post("/api/run")
    def start_run(data: dict) -> dict:
        entry = hub.get_video(str(data.get("vid")))
        selections = _selections_from_payload(data)
        job = hub.create_job(entry.vid)

        cfg = RunConfig(
            preset=str(data.get("preset") or "balanced"),
            effects=_effects_from_payload(data.get("effects") or {}),
            guard=GuardConfig(),
            encoder=str(data.get("encoder") or "auto"),
            quality=int(data.get("quality") or 60),
            work_long_side=data.get("work_long_side"),
            frame_step=int(data.get("frame_step") or 1),
            frame_start=data.get("frame_start"),
            frame_end=data.get("frame_end"),
            audio=bool(data.get("audio", True)),
            reid=bool(data.get("reid", True)),
            remove_others=bool(data.get("remove_others", False)),
            depth_blur=bool(data.get("depth_blur", False)),
        )
        if cfg.remove_others:
            cfg.clean = CleanConfig(
                segment=int(data.get("clean_segment") or 32),
                window=int(data.get("clean_window") or 16),
                dilate=int(data.get("clean_dilate") or 7),
                inpaint=bool(data.get("clean_inpaint", True)),
                inpaint_model=str(data.get("clean_inpaint_model") or "lama"),
                parallax_planes=int(data.get("clean_planes") or 3),
            )
        if not cfg.reid:
            cfg.guard.identity_floor = -1.0
            cfg.guard.reanchor_score = 2.0

        out_dir = entry.dir / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"spotlight_{int(time.time())}.mp4"

        def worker() -> None:
            job.status = "running"
            job.started = time.time()
            try:
                with hub.infer_lock:
                    if job.cancel.is_set():
                        raise Cancelled("已取消")
                    result = run(
                        entry.path, selections, out_path, models_dir, cfg,
                        on_progress=lambda p: hub.publish(job, p),
                        tmp_dir=entry.dir / "run",
                        should_cancel=job.cancel.is_set,
                    )
                job.output = Path(result.output_path)
                job.result = {
                    "n_frames": result.n_frames,
                    "elapsed": round(result.elapsed, 2),
                    "fps": round(result.frames_per_second, 2),
                    "guard": result.guard_summary,
                    "mask_stats": result.mask_stats,
                    "lost_segments": result.lost_segments[:40],
                    "mask_size": list(result.mask_size),
                    "anchors": len(selections),
                    "clean": result.clean_stats,
                }
                job.status = "done"
                hub.publish(job, Progress("done", 1, 1, "处理完成"))
                hub.finish(job, {"stage": "done", "fraction": 1.0, "message": "处理完成", **job.result})
            except Cancelled:
                job.status = "cancelled"
                job.error = ""
                hub.finish(job, {"stage": "cancelled", "fraction": 0.0, "message": "已取消"})
            except Exception as exc:  # noqa: BLE001 - 需要把任何异常回传给前端
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                hub.finish(job, {"stage": "error", "fraction": 1.0, "message": job.error})
            finally:
                job.finished = time.time()

        threading.Thread(target=worker, name=f"spotlight-{job.jid}", daemon=True).start()
        return {
            "jid": job.jid,
            "n_frames": entry.n_frames,
            "anchors": len(selections),
            "estimate": round(estimate_seconds(
                (cfg.frame_end or entry.n_frames) - (cfg.frame_start or 0),
                cfg.preset, cfg.frame_step,
            ), 1),
        }

    @app.post("/api/estimate")
    def estimate(data: dict) -> dict:
        entry = hub.get_video(str(data.get("vid")))
        preset = str(data.get("preset") or "balanced")
        step = int(data.get("frame_step") or 1)
        lo = int(data.get("frame_start") or 0)
        hi = int(data.get("frame_end") or entry.n_frames)
        n = max(0, min(entry.n_frames, hi) - max(0, lo))
        return {
            "frames": n,
            "preset": preset,
            "seconds": round(estimate_seconds(n, preset, step), 1),
        }

    @app.get("/api/progress/{jid}")
    def progress(jid: str) -> StreamingResponse:
        job = hub.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "任务不存在")
        q: queue.Queue = queue.Queue(maxsize=256)
        hub.subscribers.setdefault(jid, []).append(q)

        def stream():
            try:
                yield f"data: {json.dumps({'stage': job.status, 'fraction': job.progress.fraction, 'message': job.progress.message})}\n\n"
                while True:
                    try:
                        payload = q.get(timeout=20)
                    except queue.Empty:
                        yield ": keep-alive\n\n"
                        if job.status in ("done", "error", "cancelled"):
                            break
                        continue
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                    if payload.get("stage") in ("done", "error", "cancelled"):
                        break
            finally:
                subs = hub.subscribers.get(jid)
                if subs and q in subs:
                    subs.remove(q)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/cancel/{jid}")
    def cancel(jid: str) -> dict:
        job = hub.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "任务不存在")
        job.cancel.set()
        return {"ok": True}

    @app.get("/api/result/{jid}")
    def result(jid: str) -> FileResponse:
        job = hub.jobs.get(jid)
        if job is None or job.output is None or not job.output.is_file():
            raise HTTPException(404, "结果还没准备好")
        return FileResponse(job.output, media_type="video/mp4", filename="spotlight.mp4")

    @app.get("/api/summary/{jid}")
    def summary(jid: str) -> dict:
        job = hub.jobs.get(jid)
        if job is None:
            raise HTTPException(404, "任务不存在")
        return {
            "status": job.status, "error": job.error, **job.result,
            "has_result": bool(job.output and job.output.is_file()),
        }

    return app


def serve(host: str = "127.0.0.1", port: int = 8760, models_dir: str | Path = "models",
          open_browser: bool = True) -> None:
    import uvicorn

    app = create_app(Path(models_dir))
    url = f"http://{host}:{port}/"
    print(f"Spotlight 已启动 → {url}")
    print("按 Ctrl+C 停止。")
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
