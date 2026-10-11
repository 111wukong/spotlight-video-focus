"""Spotlight —— 视频单人聚焦：只突出你选中的那个人，其余全部虚化。

核心能力
--------
* 在任意一帧上点选或框选目标，整段视频只保留这个人的清晰与高光；
* 手持跟拍、镜头运动、目标短暂出画再回来，都尽量跟住同一个人，不中途换人；
* 全流程跑在 Apple Silicon 的 GPU 上（MLX 推理 + VideoToolbox 编码）。

快速开始
--------
::

    python -m spotlight.cli download --preset balanced
    python -m spotlight.cli serve
"""

from .appearance import AppearanceEngine, PersonBox
from .composite import EffectParams
from .pipeline import Progress, RunConfig, RunResult, Selection, preview_mask, run
from .segmenter import Segmenter, resolve_checkpoint
from .tracker import GuardConfig, GuardState, TargetGuard
from .video import FrameStore, VideoInfo, probe

__version__ = "0.1.0"

__all__ = [
    "AppearanceEngine",
    "PersonBox",
    "EffectParams",
    "GuardConfig",
    "GuardState",
    "TargetGuard",
    "Segmenter",
    "resolve_checkpoint",
    "FrameStore",
    "VideoInfo",
    "probe",
    "Selection",
    "RunConfig",
    "RunResult",
    "Progress",
    "run",
    "preview_mask",
    "__version__",
]
