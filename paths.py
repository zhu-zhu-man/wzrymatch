"""项目路径与运行环境初始化。"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
ASSETS_DIR = ROOT / "assets"
OUTPUT_DIR = ROOT / "output"
TMP_DIR = ROOT / "tmp"

# 默认示例图（可用环境变量覆盖）
DEFAULT_SCREENSHOT = ASSETS_DIR / "fighter.png"
DEFAULT_POSTER = ASSETS_DIR / "poster_new.jpg"
ALT_SCREENSHOT = ASSETS_DIR / "20260402-172820-295-dayu.jpeg"


def setup() -> Path:
    """将项目根目录加入 sys.path，并确保输出目录存在。"""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    return ROOT


def asset_path(name: str) -> Path:
    path = ASSETS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"资源不存在: {path}")
    return path


def resolve_models_dir() -> Path:
    """模型目录优先级: 环境变量 > 本地 models/ > 服务器默认路径。"""
    env_dir = os.environ.get("WZRY_MODELS_DIR")
    if env_dir:
        path = Path(env_dir)
        if path.is_dir():
            return path

    if MODELS_DIR.is_dir() and any(MODELS_DIR.glob("*.onnx")):
        return MODELS_DIR

    fallback = Path("/data/rpadata/dataset/traindata/models")
    if fallback.is_dir():
        return fallback

    return MODELS_DIR
