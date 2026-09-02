from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .capture import WindowCapture
from .config import load_config
from .window import find_window, focus_window


def screenshot(config_path: str, output: str | None) -> None:
    config = load_config(config_path)
    area = find_window(config.data["window"]["title_keywords"])
    focus_window(area.hwnd)
    capture = WindowCapture(area)
    frame = capture.grab()
    capture.close()
    path = Path(output).resolve() if output else config.resolve("../calibration/window.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)
    print(f"Saved {area.title!r} client screenshot ({area.width}x{area.height}) to:\n{path}")


def crop(config_path: str, image_path: str, name: str, threshold: float) -> None:
    config = load_config(config_path)
    image = cv2.imread(str(Path(image_path).resolve()), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    x, y, w, h = [int(v) for v in cv2.selectROI(f"Select template: {name}", image, showCrosshair=True)]
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise RuntimeError("No area was selected.")
    template_dir = config.resolve("../templates")
    template_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_dir / f"{name}.png"
    base_w, base_h = config.data["base_resolution"]
    img_h, img_w = image.shape[:2]
    crop_image = image[y : y + h, x : x + w]
    base_crop_w = max(1, round(w * base_w / img_w))
    base_crop_h = max(1, round(h * base_h / img_h))
    normalized_template = cv2.resize(crop_image, (base_crop_w, base_crop_h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(template_path), normalized_template)
    roi = [
        round(x * base_w / img_w),
        round(y * base_h / img_h),
        round(w * base_w / img_w),
        round(h * base_h / img_h),
    ]
    config.data.setdefault("detectors", {})[name] = {
        "type": "template",
        "template": f"../templates/{name}.png",
        "roi": roi,
        "threshold": threshold,
        "grayscale": True,
    }
    temp = config.path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(config.data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(config.path)
    print(f"Saved template: {template_path}")
    print(f"Updated detector {name!r} in {config.path}; base-resolution ROI={roi}")


def choose_point(config_path: str, image_path: str) -> None:
    config = load_config(config_path)
    image = cv2.imread(str(Path(image_path).resolve()), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(image_path)
    x, y, w, h = [int(v) for v in cv2.selectROI("Click-drag a tiny box around the desired point", image, showCrosshair=True)]
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise RuntimeError("No point was selected.")
    base_w, base_h = config.data["base_resolution"]
    img_h, img_w = image.shape[:2]
    point = [round((x + w / 2) * base_w / img_w), round((y + h / 2) * base_h / img_h)]
    print(f"Point in strategy coordinates: {point}")


def main() -> int:
    parser = argparse.ArgumentParser(description="BlackFlow screenshot/template calibration helper")
    sub = parser.add_subparsers(dest="command", required=True)
    shot = sub.add_parser("screenshot")
    shot.add_argument("--config", required=True)
    shot.add_argument("--output")
    crop_parser = sub.add_parser("crop")
    crop_parser.add_argument("--config", required=True)
    crop_parser.add_argument("--image", required=True)
    crop_parser.add_argument("--name", required=True)
    crop_parser.add_argument("--threshold", type=float, default=0.86)
    point_parser = sub.add_parser("point")
    point_parser.add_argument("--config", required=True)
    point_parser.add_argument("--image", required=True)
    args = parser.parse_args()
    if args.command == "screenshot":
        screenshot(args.config, args.output)
    elif args.command == "crop":
        crop(args.config, args.image, args.name, args.threshold)
    else:
        choose_point(args.config, args.image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
