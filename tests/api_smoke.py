#!/usr/bin/env python
"""网页后端接口的全链路冒烟测试。

覆盖：状态 → 上传 → 视频流 → 抽帧 → 选人预览 → 多锚点处理 → 进度流 → 摘要 → 下载，
以及取消。用真实 HTTP 客户端打真实服务，不 mock。

    python -m spotlight.cli serve --port 8761 --no-browser &
    python tests/api_smoke.py --base http://127.0.0.1:8761
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent

_pass = 0
_fail = 0


def check(name: str, ok: bool, detail: str = "") -> bool:
    global _pass, _fail
    if ok:
        _pass += 1
        print(f"  ✓ {name}" + (f"  {detail}" if detail else ""))
    else:
        _fail += 1
        print(f"  ✗ {name}" + (f"  {detail}" if detail else ""))
    return ok


def stream_until_done(client: httpx.Client, jid: str, timeout: float = 180.0):
    """读 SSE，直到出现终态。返回最后一条事件。"""
    t0 = time.time()
    last = None
    with client.stream("GET", f"/api/progress/{jid}") as s:
        for line in s.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            last = json.loads(line[6:])
            if last.get("stage") in ("done", "error", "cancelled"):
                return last
            if time.time() - t0 > timeout:
                break
    return last


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8761")
    ap.add_argument("--sample", default=str(ROOT / "samples" / "clip_boy.mp4"))
    ap.add_argument("--preset", default="turbo")
    args = ap.parse_args()

    sample = Path(args.sample)
    if not sample.is_file():
        print(f"[SKIP] 找不到测试素材 {sample}")
        print("✅ 接口冒烟跳过：0 项")
        return 0

    client = httpx.Client(base_url=args.base, timeout=240.0, trust_env=False)
    try:
        client.get("/api/status")
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] 连不上服务 {args.base}（{type(exc).__name__}）")
        print("  先启动：python -m spotlight.cli serve --port 8761 --no-browser")
        print("✅ 接口冒烟跳过：0 项")
        return 0

    try:
        r = client.get("/api/status")
        st = r.json()
        check("状态接口", r.status_code == 200 and len(st["presets"]) >= 3,
              f"{len(st['presets'])} 档，Vision {'可用' if st['vision'] else '不可用'}")
        check("档位带速度数据", all("speed" in p for p in st["presets"]),
              " ".join(f"{p['key']}={p['speed']}" for p in st["presets"]))

        with open(sample, "rb") as fh:
            r = client.post("/api/upload", files={"file": (sample.name, fh, "video/mp4")})
        info = r.json()
        check("上传视频", r.status_code == 200 and "vid" in info,
              f"{info.get('width')}x{info.get('height')} {info.get('n_frames')} 帧")
        vid = info["vid"]

        r = client.get(f"/api/video/{vid}")
        check("视频流", r.status_code == 200 and len(r.content) > 1000, f"{len(r.content)/1e6:.1f} MB")

        r = client.get(f"/api/frame/{vid}/10.jpg")
        check("抽帧", r.status_code == 200 and r.headers.get("content-type") == "image/jpeg",
              f"{len(r.content)/1024:.0f} KB")

        r = client.post("/api/select", json={
            "vid": vid, "frame_idx": 0, "points": [[0.36, 0.55]], "labels": [1], "preset": args.preset})
        j = r.json()
        check("选人预览", r.status_code == 200 and j.get("ok"),
              f"占比 {j.get('area_ratio')} · {j.get('note')}")

        r = client.post("/api/estimate", json={"vid": vid, "preset": args.preset})
        check("耗时预估", r.status_code == 200 and r.json()["seconds"] > 0,
              f"{r.json()['frames']} 帧 / {r.json()['seconds']} 秒")

        # 多锚点：主锚点 + 一个修正锚点
        anchors = [
            {"frame": 0, "points": [[0.36, 0.55]], "labels": [1]},
            {"frame": 55, "points": [[0.40, 0.60]], "labels": [1]},
        ]
        r = client.post("/api/run", json={
            "vid": vid, "anchors": anchors, "preset": args.preset,
            "effects": {"blur": 26, "brighten": 0.06, "glow": 0.25}})
        j = r.json()
        jid = j["jid"]
        check("启动多锚点处理", r.status_code == 200 and j.get("anchors") == 2, f"jid={jid}")

        t0 = time.time()
        last = stream_until_done(client, jid)
        check("处理完成", last is not None and last.get("stage") == "done",
              f"{time.time()-t0:.0f}s · {json.dumps(last, ensure_ascii=False)[:90] if last else '无事件'}")

        s = client.get(f"/api/summary/{jid}").json()
        check("摘要含掩码统计", "mask_stats" in s and s["mask_stats"]["frames_with_mask"] > 0,
              str(s.get("mask_stats")))
        check("摘要含丢失片段", "lost_segments" in s, f"{len(s.get('lost_segments', []))} 段")
        check("摘要含锚点数", s.get("anchors") == 2, str(s.get("anchors")))

        r = client.get(f"/api/result/{jid}")
        check("下载成片", r.status_code == 200 and len(r.content) > 10000, f"{len(r.content)/1e6:.1f} MB")

        # 取消：起一个长任务然后立刻取消
        r = client.post("/api/run", json={
            "vid": vid, "anchors": anchors, "preset": "balanced"})
        jid2 = r.json()["jid"]
        time.sleep(2.5)
        r = client.post(f"/api/cancel/{jid2}")
        check("取消请求被接受", r.status_code == 200, str(r.json()))
        last = stream_until_done(client, jid2, timeout=40)
        check("任务确实被取消", last is not None and last.get("stage") == "cancelled",
              json.dumps(last, ensure_ascii=False)[:90] if last else "无事件")

    finally:
        client.close()

    print()
    print(("❌" if _fail else "✅") + f" 接口冒烟：{_pass} 通过 / {_fail} 失败")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
