#!/usr/bin/env python3
"""用已知匹配参数裁剪海报（不跑特征匹配）。"""

from __future__ import annotations

import cv2

from paths import DEFAULT_POSTER, TMP_DIR, setup

setup()

# 由 match.py / whole.py 跑出的典型参数，可按实际结果修改
CENTER_X = 116.3
CENTER_Y = 263.9
SCALE = 4.94
ROTATION_ANGLE = 2.2
TARGET_SIZE = (125, 125)
BOX_W, BOX_H = 60, 56


def main() -> None:
    save_dir = TMP_DIR / "poster_cropped_rotated"
    save_dir.mkdir(parents=True, exist_ok=True)

    img_poster = cv2.imread(str(DEFAULT_POSTER))
    if img_poster is None:
        raise FileNotFoundError(f"海报不存在: {DEFAULT_POSTER}")

    h_poster, w_poster = img_poster.shape[:2]
    crop_w = int(BOX_W * SCALE)
    crop_h = int(BOX_H * SCALE)

    rot_mat = cv2.getRotationMatrix2D((CENTER_X, CENTER_Y), ROTATION_ANGLE, 1.0)
    img_rotated = cv2.warpAffine(
        img_poster,
        rot_mat,
        (w_poster, h_poster),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    x1 = max(0, int(CENTER_X - crop_w / 2))
    y1 = max(0, int(CENTER_Y - crop_h / 2))
    x2 = min(w_poster, int(CENTER_X + crop_w / 2))
    y2 = min(h_poster, int(CENTER_Y + crop_h / 2))
    cropped = img_rotated[y1:y2, x1:x2]

    raw_path = save_dir / "cropped_raw_original.png"
    final_path = save_dir / "cropped_125x125.png"
    cv2.imwrite(str(raw_path), cropped)
    cv2.imwrite(str(final_path), cv2.resize(cropped, TARGET_SIZE, interpolation=cv2.INTER_AREA))

    print(f"裁剪原图: {raw_path}")
    print(f"最终头像: {final_path}")


if __name__ == "__main__":
    main()
