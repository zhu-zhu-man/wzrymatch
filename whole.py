#!/usr/bin/env python3
"""完整流程：截图检测 -> 匹配 -> 裁剪海报头像。"""

from __future__ import annotations

import traceback

import cv2
import numpy as np

from paths import DEFAULT_POSTER, DEFAULT_SCREENSHOT, OUTPUT_DIR, setup

setup()

from lllcv.detection import Manual_Detection
from lllcv.image_composer import ImageComposer
from lllcv.matching import SIFT_Matching

# ===================== 配置 =====================
GAME_SCREENSHOT_PATH = DEFAULT_SCREENSHOT
POSTER_PATH = DEFAULT_POSTER

RENDER_SAVE_PATH = OUTPUT_DIR / "rendered.png"
CROP_RAW_SAVE_PATH = OUTPUT_DIR / "cropped_raw.png"
CROP_FINAL_SAVE_PATH = OUTPUT_DIR / "cropped_125x125.png"

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
TARGET_SIZE = (125, 125)
# ==================================================


def main() -> None:
    print("\n==================== 完整匹配裁剪 ====================")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    matcher = SIFT_Matching()
    img_origin = cv2.imread(str(GAME_SCREENSHOT_PATH))
    if img_origin is None:
        raise FileNotFoundError(f"截图不存在: {GAME_SCREENSHOT_PATH}")

    h1, w1 = img_origin.shape[:2]
    GRID_CONFIG["w1"] = w1
    GRID_CONFIG["h1"] = h1

    griddetection = Manual_Detection()
    grids = griddetection.detect(img_origin, GRID_CONFIG)

    rendered = img_origin.copy()
    for grid in grids:
        x, y, w_box, h_box = grid["rect"]
        cv2.rectangle(rendered, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    cv2.imwrite(str(RENDER_SAVE_PATH), rendered)
    print(f"检测框图: {RENDER_SAVE_PATH}")

    rect = grids[HEAD_INDEX]["rect"]
    img_search = img_origin[rect[1] : rect[1] + rect[3], rect[0] : rect[0] + rect[2]]

    composer = ImageComposer()
    composer.set_image_adjust("mask_circle", {"ratio": 0.85})
    img_search = composer.compose(img_search)

    img_poster = cv2.imread(str(POSTER_PATH))
    if img_poster is None:
        raise FileNotFoundError(f"海报不存在: {POSTER_PATH}")

    cx, cy, scale, rotation, mean_err, inliers = matcher.match_image(img_search, img_poster)
    print(f"匹配: 中心=({cx:.1f},{cy:.1f}) 缩放={scale:.2f} 旋转={rotation:.1f}°")

    h_poster, w_poster = img_poster.shape[:2]
    crop_w = int(GRID_CONFIG["w"] * scale)
    crop_h = int(GRID_CONFIG["h"] * scale)

    rot_mat = cv2.getRotationMatrix2D((cx, cy), rotation, 1.0)
    img_rotated = cv2.warpAffine(
        img_poster,
        rot_mat,
        (w_poster, h_poster),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    x1 = max(0, int(cx - crop_w / 2))
    y1 = max(0, int(cy - crop_h / 2))
    x2 = min(w_poster, int(cx + crop_w / 2))
    y2 = min(h_poster, int(cy + crop_h / 2))
    cropped = img_rotated[y1:y2, x1:x2]

    cv2.imwrite(str(CROP_RAW_SAVE_PATH), cropped)
    img_final = cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(CROP_FINAL_SAVE_PATH), img_final)

    print(f"裁剪原图: {CROP_RAW_SAVE_PATH}")
    print(f"最终头像: {CROP_FINAL_SAVE_PATH}")
    print("完成")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
