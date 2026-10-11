#!/usr/bin/env python
"""核心行为的回归测试（不需要浏览器，跑得快）。

覆盖的是那些「改错了不会立刻报错、但上线就出事」的地方：

* 掩码存储不占进程内存（曾经每帧留内存，300 秒素材要 8.3 GB）
* 掩码读写的正确性（改过一次底层 I/O，写进去和读出来必须一致）
* 取消能真正中断（曾经只在开始前检查一次，跑起来后按取消没用）
* 帧范围只处理指定区间，区间外原样输出
* 多锚点：修正锚点要能生效，且不能被守卫覆盖
* 碎块清理：只保留主要连通域，且不能误伤主掩码
* 清除旁人：合成场景下有 ground truth，验证重建精度、截尾均值不劣化、取样窗口被遵守

    python tests/test-core.py
"""

from __future__ import annotations

import os
import resource
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from spotlight.pipeline import (  # noqa: E402
    Cancelled,
    RunConfig,
    Selection,
    _clean_specks,
    estimate_seconds,
    run,
)
from spotlight.video import MaskStore, gate_levels, lost_segments, summarize_masks  # noqa: E402

MODELS = str(ROOT / "models")
SAMPLE = str(ROOT / "samples" / "c855564.mp4")

_pass = 0
_fail = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _pass, _fail
    if ok:
        _pass += 1
        print(f"  ✓ {name}" + (f"  {detail}" if detail else ""))
    else:
        _fail += 1
        print(f"  ✗ {name}" + (f"  {detail}" if detail else ""))


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


# ---------------------------------------------------------------- 掩码存储


def test_mask_store() -> None:
    print("掩码存储")
    tmp = tempfile.mkdtemp()
    try:
        base = rss_mb()
        for n in (200, 4000):
            st = MaskStore(os.path.join(tmp, f"m{n}.u8"), n, 640, 360)
            row = np.full((360, 640), 90, np.uint8)
            for i in range(n):
                row[0, 0] = i % 255
                st.set(i, row)
            st.finalize()
            back = st.get(n // 2)
            ok = back.shape == (360, 640) and int(back[0, 0]) == (n // 2) % 255
            check(f"{n} 帧读写一致", ok)
            st.close()
        grown = rss_mb() - base
        # 4000 帧 640x360 是 0.92 GB 数据；进程内存不该跟着涨
        check("掩码不占进程内存", grown < 60, f"峰值只涨了 {grown:.0f} MB")

        # 空洞补齐
        st = MaskStore(os.path.join(tmp, "gap.u8"), 10, 64, 64)
        a = np.full((64, 64), 10, np.uint8)
        b = np.full((64, 64), 200, np.uint8)
        st.set(0, a)
        st.set(9, b)
        st.finalize(interpolate=True)
        mid = int(st.get(5)[0, 0])
        check("空洞线性插值", 90 <= mid <= 120, f"第 5 帧 = {mid}（期望 10~200 的中间值）")
        st.close()

        # 存在门与丢失片段
        present = np.array([True] * 5 + [False] * 6 + [True] * 4)
        levels = gate_levels(present, ramp_up=3, ramp_down=7)
        check("目标在画面里时门是满的", levels[3] == 255, f"第 3 帧 = {levels[3]}")
        check("离开画面时门在衰减", levels[5] < 255 and levels[10] < levels[5],
              f"第 5 帧 {levels[5]} → 第 10 帧 {levels[10]}")
        check("回到画面时门恢复", levels[-1] == 255, f"末帧 = {levels[-1]}")
        long_gone = gate_levels(np.array([True] + [False] * 20), ramp_down=7)
        check("离开足够久后门归零", long_gone[-1] == 0, f"末帧 = {long_gone[-1]}")
        segs = lost_segments(present, min_len=3)
        check("丢失片段识别", len(segs) == 1 and segs[0] == {"start": 5, "end": 10, "frames": 6},
              str(segs))
        stats = summarize_masks(np.array([0, 100, 200, 300]), np.array([False, True, True, True]))
        check("掩码统计", stats["frames_with_mask"] == 3 and stats["area_median"] == 200, str(stats))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- 碎块清理


def test_clean_specks() -> None:
    print("\n碎块清理")
    mask = np.zeros((360, 640), np.uint8)
    mask[100:300, 200:400] = 255          # 主体
    mask[20:24, 30:34] = 255              # 一个 4x4 的杂散小块
    out = _clean_specks(mask, 0.08)
    body_kept = int((out[100:300, 200:400] > 100).sum())
    speck_kept = int((out[20:24, 30:34] > 100).sum())
    check("主体被保留", body_kept > 0.95 * 200 * 200, f"{body_kept} / {200*200}")
    check("杂散小块被清掉", speck_kept == 0, f"残留 {speck_kept} 像素")

    single = np.zeros((360, 640), np.uint8)
    single[50:200, 100:300] = 255
    same = _clean_specks(single, 0.08)
    check("只有一块时原样返回", np.array_equal(single, same))

    check("min_ratio=0 时直接跳过", np.array_equal(mask, _clean_specks(mask, 0.0)))


# ---------------------------------------------------------------- 流水线行为


def _base_cfg(preset: str = "turbo", **kw) -> RunConfig:
    cfg = RunConfig(preset=preset, **kw)
    cfg.guard.identity_floor = 0.45
    return cfg


def test_cancel() -> None:
    print("\n取消")
    flag = threading.Event()
    threading.Timer(4.0, flag.set).start()
    t0 = time.time()
    try:
        run(SAMPLE, Selection(0, [(0.55, 0.65)], [1]), "/tmp/_spotlight_cancel.mp4", MODELS,
            _base_cfg(), should_cancel=flag.is_set)
        check("能被打断", False, "跑完了，没有中断")
    except Cancelled:
        dt = time.time() - t0
        check("能被打断", dt < 12, f"{dt:.1f} 秒中断（4 秒时请求）")
    except Exception as exc:  # noqa: BLE001
        check("能被打断", False, f"抛了别的异常 {type(exc).__name__}: {exc}")


def test_frame_range() -> None:
    print("\n帧范围")
    res = run(SAMPLE, Selection(0, [(0.55, 0.65)], [1]), "/tmp/_spotlight_range.mp4", MODELS,
              _base_cfg(frame_start=0, frame_end=60))
    check("只处理指定区间", res.mask_stats["frames_with_mask"] <= 62,
          f"有掩码的帧 {res.mask_stats['frames_with_mask']}（区间 0~59）")
    check("区间外报为丢失", any(s["start"] >= 60 for s in res.lost_segments),
          str([(s["start"], s["end"]) for s in res.lost_segments]))


def test_multi_anchor() -> None:
    print("\n多锚点")
    res = run(
        SAMPLE,
        [Selection(0, [(0.55, 0.65)], [1]), Selection(80, [(0.52, 0.62)], [1])],
        "/tmp/_spotlight_multi.mp4", MODELS, _base_cfg(frame_start=0, frame_end=140),
    )
    check("跑通", res.mask_stats["frames_with_mask"] > 100,
          f"有掩码的帧 {res.mask_stats['frames_with_mask']}")
    # 修正锚点那一帧必须真的有掩码（不能被守卫覆盖成空）
    check("锚点帧有掩码", res.mask_stats["area_median"] > 0, f"中位面积 {res.mask_stats['area_median']}")
    # 区间外（>=140）本来就不处理，会被算成「丢失」，这里只检查已处理区间内没有长空洞
    inner = [s for s in res.lost_segments if s["start"] < 140 and s["frames"] > 20]
    check("已处理区间内没有长空洞", not inner, str([(s["start"], s["end"]) for s in inner]))


def _synth_scene(
    path: Path, n: int = 48, w: int = 320, h: int = 180, shift: float = 26.0,
    red_outside: tuple[int, int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """造一段「背景有纹理、镜头在平移、有一个人走过」的合成素材。

    返回 ``(人掩码序列, 真实背景序列)``。**有了真实背景，就能精确度量重建误差**——
    这是判断「残影有多重」的唯一可靠办法（真实素材没有 ground truth）。

    ``red_outside`` 给定时，把落在 ``(lo, hi)`` 之外的帧整帧涂成纯红。
    用来验证「取样窗口真的被遵守了」：如果窗口外那些红帧混进了板里，
    板就会偏红。
    """
    import cv2

    from spotlight.video import write_frame_video

    rng = np.random.default_rng(5)
    bg = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
    bg = cv2.GaussianBlur(bg, (0, 0), 3)
    for _ in range(40):
        x, y = (int(v) for v in rng.integers(0, min(w, h), size=2))
        cv2.rectangle(bg, (x, y), (min(w, x + 26), min(h, y + 26)),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)

    frames, pmasks, truth = [], [], []
    for i in range(n):
        shift_x = shift * np.sin(i / n * np.pi)
        M = np.float32([[1, 0, shift_x], [0, 1, 0]])
        clean_bg = cv2.warpAffine(bg, M, (w, h), flags=cv2.INTER_LINEAR)
        frame = clean_bg.copy()
        pm = np.zeros((h, w), np.uint8)
        px = int((w * 0.1 + i * 4) % (w + 80) - 40)
        cv2.ellipse(frame, (px, int(h * 0.55)), (13, 30), 0, 0, 360, (250, 30, 30), -1)
        cv2.ellipse(pm, (px, int(h * 0.55)), (15, 32), 0, 0, 360, 255, -1)
        if red_outside is not None and not (red_outside[0] <= i < red_outside[1]):
            frame[:] = (255, 0, 0)
        frames.append(frame)
        pmasks.append(pm)
        truth.append(clean_bg)

    write_frame_video(iter(frames), path, w, h, "30/1", total=n, encoder="ffv1")
    return pmasks, truth


def _plate_error(store, persons, homographies, plates, truth, ref, size) -> float:
    """板映射回第 ``ref`` 帧后，与人像区真实背景的平均绝对差。"""
    import cv2

    from spotlight.cleanplate import CleanConfig, PeopleRemover

    rm = PeopleRemover(CleanConfig(inpaint=False), None)
    plate = rm._background_for(ref, plates, homographies, size)
    tb = cv2.resize(truth[ref], size)
    pm = persons.get(ref) > 127
    if not pm.any():
        return -1.0
    return float(np.abs(plate.astype(np.int16) - tb.astype(np.int16)).mean(axis=2)[pm].mean())


def test_cleanplate() -> None:
    """清除旁人：背景板重建的正确性。"""
    import cv2

    from spotlight.cleanplate import CleanConfig, PeopleRemover, _frame_homographies
    from spotlight.video import FrameStore, MaskStore, write_frame_video

    print("\n清除旁人（合成场景，有 ground truth）")
    work = Path(tempfile.mkdtemp(prefix="spotlight-clean-"))
    try:
        video = work / "synth.mkv"
        pmasks, truth = _synth_scene(video)
        store = FrameStore(str(video), work / "frames", long_side=None)
        persons = MaskStore(work / "persons.u8", store.n_frames, store.width, store.height)
        for i, m in enumerate(pmasks):
            persons.set(i, cv2.resize(m, (store.width, store.height), interpolation=cv2.INTER_NEAREST))
        homographies = _frame_homographies(store, persons, CleanConfig(), None)
        size = (store.width, store.height)

        # 1) 重建精度：板上「被抹掉的位置」应当接近真实背景
        rm = PeopleRemover(CleanConfig(segment=32, window=16, inpaint=False), None)
        plates, holes = rm.build_plates(store, persons, homographies, frame_range=(0, store.n_frames))
        err = _plate_error(store, persons, homographies, plates, truth, 24, size)
        check("重建误差足够小", 0 < err < 20, f"人像区平均误差 {err:.1f} 灰阶")

        # 2) 截尾均值不能把结果改坏
        rm_plain = PeopleRemover(CleanConfig(segment=32, window=16, inpaint=False, robust_tol=0.0), None)
        plates_plain, _ = rm_plain.build_plates(store, persons, homographies, frame_range=(0, store.n_frames))
        err_plain = _plate_error(store, persons, homographies, plates_plain, truth, 24, size)
        check("截尾均值不劣化", err <= err_plain + 0.5,
              f"截尾 {err:.2f} vs 普通 {err_plain:.2f}")

        # 3) 取样窗口必须被遵守：窗口外全是纯红，板不能被染红。
        #    这是「视差与空洞的平衡」那个旋钮的语义保证——窗口开多大就只采多大范围。
        red_video = work / "red.mkv"
        red_persons, red_truth = _synth_scene(red_video, n=96, red_outside=(30, 56))
        rstore = FrameStore(str(red_video), work / "rframes", long_side=None)
        rpersons = MaskStore(work / "rpersons.u8", rstore.n_frames, rstore.width, rstore.height)
        for i, m in enumerate(red_persons):
            rpersons.set(i, cv2.resize(m, (rstore.width, rstore.height), interpolation=cv2.INTER_NEAREST))
        rh = _frame_homographies(rstore, rpersons, CleanConfig(), None)
        rh_, rw_ = rstore.height, rstore.width

        def redness(lo: int, hi: int, center: int) -> float:
            """板在「真实背景明显不红」的中央区域里，红-绿通道差（越大越被红污染）。"""
            r = PeopleRemover(CleanConfig(inpaint=False), None)
            plate, _ = r._build_plate(rstore, rpersons, rh, lo, hi, reference=center,
                                      sigma=max(2.0, (hi - lo) / 4.0))
            tb = cv2.resize(red_truth[center], (rw_, rh_))
            not_red_bg = (tb[:, :, 0].astype(np.int16) - tb[:, :, 1].astype(np.int16)) < 60
            roi = np.zeros_like(not_red_bg)
            roi[int(rh_ * 0.25):int(rh_ * 0.75), int(rw_ * 0.25):int(rw_ * 0.75)] = True
            sel = not_red_bg & roi
            if not sel.any():
                return -1.0
            return float(plate[:, :, 0].astype(np.int16)[sel].mean()
                         - plate[:, :, 1].astype(np.int16)[sel].mean())

        tight = redness(34, 51, 42)   # 窗口完全落在非红区间 [30, 56) 内
        wide = redness(2, 83, 42)     # 窗口把红帧也圈进来了
        check("取样窗口被遵守（窗口外的内容进不了板）", tight < 25,
              f"窄窗红偏 {tight:.0f}")
        check("该断言真的能失败（宽窗确实会被污染）", wide > tight + 40,
              f"宽窗红偏 {wide:.0f} vs 窄窗 {tight:.0f}")

        store.close()
        rstore.close()
    finally:
        shutil.rmtree(work, ignore_errors=True)


def test_diffusion_shape() -> None:
    """修复 diffusers 输出 512 方图时与横屏源图广播失败的回归。"""
    from types import SimpleNamespace
    from PIL import Image
    from spotlight.inpaint_diffusion import DiffusionInpainter

    class FixedSquarePipeline:
        def __init__(self):
            self.kwargs = None

        def __call__(self, **kwargs):
            self.kwargs = kwargs
            # 模拟模型无视传入宽高、仍回默认 512x512 的情况。
            return SimpleNamespace(images=[Image.new("RGB", (512, 512), (240, 30, 20))])

    inpainter = DiffusionInpainter.__new__(DiffusionInpainter)
    inpainter.available = True
    inpainter._pipe = FixedSquarePipeline()
    inpainter.steps = 28
    inpainter.size = 1024
    inpainter.guidance_scale = 5.5
    inpainter.context = 64
    inpainter.seed = 1731

    print("\n扩散补洞分辨率与画质参数")
    rgb = np.full((540, 960, 3), 80, dtype=np.uint8)
    hole = np.zeros((540, 960), dtype=np.uint8)
    hole[180:360, 350:610] = 255
    out = inpainter.fill(rgb, hole)
    outside = hole == 0
    inside = hole > 127
    check("横屏补洞输出尺寸匹配", out.shape == rgb.shape, f"{out.shape}")
    kwargs = inpainter._pipe.kwargs
    check("局部裁剪保留洞区上下文", kwargs["image"].size == (384, 304),
          str(kwargs["image"].size))
    check("提示词约束场景连续与无人", "seamless continuation" in kwargs["prompt"]
          and "person" in kwargs["negative_prompt"])
    check("扩散引导与步数提高", kwargs["guidance_scale"] == 5.5
          and kwargs["num_inference_steps"] == 28)
    check("横屏洞外像素保持原样", np.array_equal(out[outside], rgb[outside]))
    check("横屏洞内确实被替换", not np.array_equal(out[inside], rgb[inside]))

    # 同时守住调用方传入 mask 尺寸意外不一致的情况。
    square = np.full((512, 512, 3), 80, dtype=np.uint8)
    mismatched_hole = np.zeros((540, 960), dtype=np.uint8)
    mismatched_hole[180:360, 350:610] = 255
    normalized = inpainter.fill(square, mismatched_hole)
    check("异尺寸掩码不会触发广播错误", normalized.shape == square.shape, str(normalized.shape))


def test_estimate() -> None:
    print("\n耗时预估")
    a = estimate_seconds(1000, "balanced")
    b = estimate_seconds(1000, "turbo")
    check("档位越快估得越短", b < a, f"turbo {b:.0f}s < balanced {a:.0f}s")
    check("单调递增", estimate_seconds(2000, "balanced") > a)


def main() -> int:
    test_diffusion_shape()
    if not Path(SAMPLE).is_file():
        print(f"[SKIP] 找不到测试素材 {SAMPLE}")
        print("✅ 核心测试跳过：0 项")
        return 0
    if not Path(MODELS).is_dir() or not any(Path(MODELS).rglob("*.safetensors")):
        print(f"[SKIP] 找不到模型权重 {MODELS}")
        print("✅ 核心测试跳过：0 项")
        return 0

    test_mask_store()
    test_clean_specks()
    test_cleanplate()
    test_estimate()
    test_cancel()
    test_frame_range()
    test_multi_anchor()

    print()
    print(("❌" if _fail else "✅") + f" 核心测试：{_pass} 通过 / {_fail} 失败")
    return 1 if _fail else 0


if __name__ == "__main__":
    sys.exit(main())
