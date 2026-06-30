#!/usr/bin/env python3
"""头像检测 + SIFT/DeDoDe 匹配（仅输出匹配参数）。"""

from __future__ import annotations

import os
import traceback

import cv2

from paths import DEFAULT_POSTER, DEFAULT_SCREENSHOT, TMP_DIR, setup

setup()

from lllcv.detection import Manual_Detection
from lllcv.image_composer import ImageComposer
from lllcv.matching import SIFT_Matching

# ===================== 配置 =====================
GAME_SCREENSHOT_PATH = DEFAULT_SCREENSHOT
POSTER_PATH = DEFAULT_POSTER

GRID_CONFIG = {
    "type": "grid",
    "x0": 581,
    "y0": 15,
    "w": 60,
    "h": 60,
    "n": 4,
    "m": 1,
    "dx": 80,
    "dy": 0,
    "w0": 2778,
    "h0": 1284,
}
HEAD_INDEX = 0
# ==================================================


def main() -> None:
    print("\n==================== 头像匹配 ====================")

    matcher = SIFT_Matching()

    img_origin = cv2.imread(str(GAME_SCREENSHOT_PATH))
    if img_origin is None:
        raise FileNotFoundError(f"截图不存在: {GAME_SCREENSHOT_PATH}")

    h1, w1 = img_origin.shape[:2]
    GRID_CONFIG["w1"] = w1
    GRID_CONFIG["h1"] = h1
    print(f"原图尺寸: {w1}x{h1}")

    griddetection = Manual_Detection()
    grids = griddetection.detect(img_origin, GRID_CONFIG)
    print(f"检测到 {len(grids)} 个头像区域")

    rendered = img_origin.copy()
    for grid in grids:
        x, y, w_box, h_box = grid["rect"]
        cv2.rectangle(rendered, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)

    render_path = TMP_DIR / "rendered.png"
    cv2.imwrite(str(render_path), rendered)
    print(f"检测框图: {render_path}")

    rect = grids[HEAD_INDEX]["rect"]
    img_search = img_origin[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

    composer = ImageComposer()
    composer.set_image_adjust("mask_circle", {"ratio": 0.85})
    img_search = composer.compose(img_search)

    img_src = cv2.imread(str(POSTER_PATH))
    if img_src is None:
        raise FileNotFoundError(f"海报不存在: {POSTER_PATH}")

    cx, cy, scale, rotation, mean_err, inliers = matcher.match_image(img_search, img_src)
    print("匹配成功")
    print(f"  中心: ({cx:.1f}, {cy:.1f})")
    print(f"  缩放: {scale:.2f}")
    print(f"  旋转: {rotation:.1f}°")
    print(f"  内点: {inliers}, 误差: {mean_err:.2f}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
