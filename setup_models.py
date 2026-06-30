#!/usr/bin/env python3
"""将运行所需 ONNX 模型复制到本地 models/ 目录。"""

from __future__ import annotations

import shutil
from pathlib import Path

from paths import MODELS_DIR, ROOT, setup

# SIFT 匹配流程最少依赖的模型
REQUIRED_MODELS = [
    "dedode_end2end_1024_fp16.onnx",
]

FALLBACK_DIR = Path("/data/rpadata/dataset/traindata/models")


def main() -> int:
    setup()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    missing = []
    for name in REQUIRED_MODELS:
        dst = MODELS_DIR / name
        if dst.exists():
            print(f"已存在: {dst}")
            continue

        src = FALLBACK_DIR / name
        if not src.exists():
            missing.append(name)
            print(f"缺失且无法复制: {name} (源路径不存在: {src})")
            continue

        print(f"复制: {src} -> {dst}")
        shutil.copy2(src, dst)

    if missing:
        print("\n请手动将以下模型放入:", MODELS_DIR)
        for name in missing:
            print(f"  - {name}")
        return 1

    print(f"\n模型目录就绪: {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
