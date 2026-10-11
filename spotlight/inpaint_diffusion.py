"""扩散模型补洞（SD inpainting，基于 diffusers）。

为什么是「质量」档才用的补洞模型
--------------------------------
清除旁人的空洞由 :class:`~spotlight.cleanplate.PeopleRemover` 在**背景板**上补，
而背景板是「每段一张、段内共用」，所以补洞成本按**段数**摊销，不是逐帧——
一部短视频也就 5~9 张板，因此即便扩散模型慢，也只多花几分钟，完全可接受。

它用来替代 :class:`~spotlight.cleanplate.LamaInpainter`（LaMa，CNN，快但补出来的
内容是「合理但不对」的）。扩散模型在有纹理、有结构的背景上明显更真：砖墙、地砖、
树叶、人群后方的复杂场景，LaMa 容易糊成一团，扩散模型能长出对的纹理。

依赖
----
需要 ``torch`` + ``diffusers`` + ``Pillow``（HF 当前网络受限时装不上，装好且下载
权重后即可启用）。全部 **lazy import**，不启用扩散时不会触碰这些重依赖，也不会影响
默认的 LaMa 路径。

默认走 LaMa；只有 ``CleanConfig.inpaint_model == "diffusion"`` 且权重就绪时才用它。
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from .composite import VideoError  # noqa: F401  (复用异常类型，便于上层识别)

ProgressFn = Callable[..., None]


def resolve_diffusion(models_dir: str | Path) -> Path | None:
    """找扩散补洞模型的目录（diffusers 格式）。

    要求关键权重文件齐全，避免把中途中断/空目录误判为已就绪。
    """
    cand = Path(models_dir) / "diffusion"
    if not cand.is_dir():
        return None
    required = (
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer_config.json",
        "unet/config.json",
        "vae/config.json",
    )
    if any(not (cand / rel).is_file() or (cand / rel).stat().st_size == 0 for rel in required):
        return None
    # Treat partially streamed multi-GB safetensors as unavailable, not ready.
    minimum_sizes = {
        "safety_checker/model.safetensors": 608_018_446,
        "text_encoder/model.safetensors": 246_144_867,
        "unet/diffusion_pytorch_model.safetensors": 1_719_154_071,
        "vae/diffusion_pytorch_model.safetensors": 167_335_315,
    }
    for rel, size in minimum_sizes.items():
        path = cand / rel
        if not path.is_file() or path.stat().st_size < size:
            return None
    return cand


class DiffusionInpainter:
    """Stable Diffusion Inpainting（diffusers），作为 LaMa 的「质量」替代。

    接口与 :class:`~spotlight.cleanplate.LamaInpainter` 完全一致：``fill(rgb, hole)``
    把 ``hole``（非零=要补）区域补上，返回补好的图；失败时原样返回。
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str | None = None,
        steps: int = 28,
        size: int = 1024,
        guidance_scale: float = 5.5,
        context: int = 64,
        seed: int = 1731,
    ) -> None:
        self.model_path = Path(model_path)
        self.available = False
        self.reason = ""
        self._pipe = None
        self.steps = int(steps)
        self.size = int(size)
        self.guidance_scale = float(guidance_scale)
        self.context = max(0, int(context))
        self.seed = int(seed)
        self.device = "cpu"

        if not self.model_path.is_dir():
            self.reason = f"模型目录不存在：{self.model_path}"
            return
        try:
            import torch  # noqa: F401
            from diffusers import StableDiffusionInpaintPipeline

            dev = device or ("mps" if _mps_available() else "cpu")
            self.device = dev
            dtype = torch.float32  # 兼容性优先；MPS 上 float16 的 inpaint 偶发不稳
            self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(self.model_path), torch_dtype=dtype
            ).to(dev)
            # 背景板没有 NSFW 风险，关掉安全审查省一次子模型加载
            try:
                self._pipe.safety_checker = None  # type: ignore[assignment]
            except Exception:  # noqa: BLE001
                pass
            self.available = True
        except Exception as exc:  # noqa: BLE001 - 装不上/加载失败都应让上层给出清晰提示
            self.reason = f"{type(exc).__name__}: {exc}"

    def fill(self, rgb, hole):
        """把 ``hole`` 区域用扩散模型补上。失败时原样返回。"""
        if not self.available or self._pipe is None:
            return rgb
        import numpy as np
        import cv2
        from PIL import Image

        rgb = np.asarray(rgb)
        hole = np.asarray(hole)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb 应为 HxWx3，实际形状 {rgb.shape}")
        if hole.ndim != 2:
            raise ValueError(f"hole 应为 HxW，实际形状 {hole.shape}")
        h, w = rgb.shape[:2]
        # 边界处可能收到不同分辨率的掩码；统一回原图尺寸再做推理与合成。
        if hole.shape != (h, w):
            hole = cv2.resize(hole, (w, h), interpolation=cv2.INTER_NEAREST)
        hole = np.where(hole > 127, 255, 0).astype(np.uint8)
        ys, xs = np.where(hole > 0)
        if not len(xs):
            return rgb

        # 围绕洞区裁一块带上下文的局部图。以前整张 1080p 板缩至 1024px 后，
        # 小洞只剩几十个像素；局部裁剪能把模型算力集中到缺失区域，同时保留周边纹理。
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        context = max(self.context, int(round(max(x1 - x0, y1 - y0) * 0.2)))
        x0, y0 = max(0, x0 - context), max(0, y0 - context)
        x1, y1 = min(w, x1 + context), min(h, y1 + context)
        crop = rgb[y0:y1, x0:x1]
        crop_hole = hole[y0:y1, x0:x1]
        ch, cw = crop.shape[:2]

        # 把局部区域限制到 size 长边，并对齐到 SD 要求的 8 像素倍数。
        scale = min(1.0, max(8, int(self.size)) / max(ch, cw))
        proc_w = max(8, int(round(cw * scale)) // 8 * 8)
        proc_h = max(8, int(round(ch * scale)) // 8 * 8)
        if (proc_h, proc_w) != (ch, cw):
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
            img = cv2.resize(crop, (proc_w, proc_h), interpolation=interp)
            msk = cv2.resize(crop_hole, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
        else:
            img, msk = crop, crop_hole

        img_p = Image.fromarray(img)          # RGB
        msk_p = Image.fromarray(msk)          # 白=要补
        prompt = (
            "photorealistic seamless continuation of the existing background, "
            "match surrounding scene, perspective, colors, texture and lighting, "
            "natural realistic details, empty background"
        )
        negative_prompt = (
            "person, people, human, face, body, silhouette, crowd, duplicate objects, "
            "text, watermark, blurry, distorted, low quality"
        )
        try:
            import torch
            # 固定噪声使同一背景板重复处理可复现，减少不同段补洞的随机风格跳变。
            generator = torch.Generator(device="cpu").manual_seed(self.seed)
            out = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=img_p,
                mask_image=msk_p,
                height=proc_h,
                width=proc_w,
                num_inference_steps=self.steps,
                guidance_scale=self.guidance_scale,
                strength=1.0,
                generator=generator,
                output_type="pil",
            ).images[0]
        except Exception:  # noqa: BLE001 - 单张失败不应中断整段
            return rgb

        out_rgb = np.array(out.convert("RGB"))
        # diffusers/个别模型仍可能回退到默认 512x512；强制校准，禁止广播时报错或拉伸掩码。
        if out_rgb.shape[:2] != (proc_h, proc_w):
            out_rgb = cv2.resize(out_rgb, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR)
        if (proc_h, proc_w) != (ch, cw):
            out_rgb = cv2.resize(out_rgb, (cw, ch), interpolation=cv2.INTER_LINEAR)
        # 只在局部洞内做软边羽化，洞外像素保持原样。
        alpha = cv2.GaussianBlur(crop_hole, (0, 0), 2.0).astype(np.float32) / 255.0
        alpha *= (crop_hole > 0).astype(np.float32)
        alpha = alpha[..., None]
        result = rgb.copy()
        result[y0:y1, x0:x1] = (
            crop.astype(np.float32) * (1 - alpha) + out_rgb.astype(np.float32) * alpha
        ).astype(np.uint8)
        return result


def _mps_available() -> bool:
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except Exception:  # noqa: BLE001
        return False
