#!/usr/bin/env python3
"""Run RF-DETR segmentation inference on images."""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
from PIL import Image
from rfdetr import RFDETRSegNano, RFDETRSegSmall


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _contrast_stretch(img: np.ndarray, low_p: float, high_p: float, per_channel: bool) -> np.ndarray:
    """Percentile-based contrast stretching. Matches Roboflow's auto-adjust contrast."""
    img = img.astype(np.float32)
    result = np.empty_like(img)

    if per_channel and img.ndim == 3:
        for c in range(img.shape[2]):
            channel = img[:, :, c]
            p_low = np.percentile(channel, low_p)
            p_high = np.percentile(channel, high_p)
            if p_high > p_low:
                result[:, :, c] = np.clip((channel - p_low) / (p_high - p_low) * 255.0, 0, 255)
            else:
                result[:, :, c] = channel
    else:
        p_low = np.percentile(img, low_p)
        p_high = np.percentile(img, high_p)
        if p_high > p_low:
            result = np.clip((img - p_low) / (p_high - p_low) * 255.0, 0, 255)
        else:
            result = img.copy()

    return result.astype(np.uint8)


_SIZE_TO_CLASS = {"nano": RFDETRSegNano, "small": RFDETRSegSmall}


def load_model(model_path: str):
    name = Path(model_path).stem.lower()
    cls = next((v for k, v in _SIZE_TO_CLASS.items() if k in name), None)
    if cls is None:
        raise ValueError(f"Cannot infer model size from '{model_path}'. Expected 'nano' or 'small' in filename.")
    return cls(pretrain_weights=model_path, num_classes=2, device="cuda")


def annotate(image_bgr: np.ndarray, detections: sv.Detections, class_names: list[str]) -> np.ndarray:
    annotated = image_bgr.copy()
    if detections.mask is not None:
        annotated = sv.MaskAnnotator().annotate(annotated, detections)
    annotated = sv.BoxAnnotator().annotate(annotated, detections)
    if class_names and detections.class_id is not None:
        labels = [class_names[i] if i < len(class_names) else str(i) for i in detections.class_id]
    else:
        labels = None
    annotated = sv.LabelAnnotator().annotate(annotated, detections, labels=labels)
    return annotated


def run_on_image(model, image_path: Path, output_dir: Path | None,
                 threshold: float, low_p: float, high_p: float, per_channel: bool,
                 class_names: list[str]):
    image_bgr = cv2.imread(str(image_path))
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = _contrast_stretch(image_rgb, low_p, high_p, per_channel)

    t0 = time.perf_counter()
    detections = model.predict(Image.fromarray(image_rgb), threshold=threshold)
    infer_fps = 1.0 / (time.perf_counter() - t0)
    print(f"{image_path.name}: {len(detections)} detection(s)  inference {infer_fps:.1f} Hz")

    annotated = annotate(image_bgr, detections, class_names)

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / image_path.name), annotated)
    else:
        cv2.imshow(image_path.name, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="Path to .pt model file")
    p.add_argument("input", help="Image file or directory of images")
    p.add_argument("-o", "--output", default=None, help="Save annotated images here instead of displaying")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--low-p", type=float, default=2.0, help="Low percentile for contrast stretch")
    p.add_argument("--high-p", type=float, default=98.0, help="High percentile for contrast stretch")
    p.add_argument("--per-channel", action="store_true", default=True, help="Apply contrast stretch per channel")
    p.add_argument("--classes", nargs="+", default=["front", "side"], help="Class names (in order)")
    args = p.parse_args()

    t0 = time.perf_counter()
    model = load_model(args.model)
    model.optimize_for_inference()
    load_ms = (time.perf_counter() - t0) * 1000
    print(f"Model loaded in {load_ms:.0f} ms")

    # warmup — first CUDA call compiles kernels, skew timing otherwise
    from PIL import Image as _PIL
    _dummy = _PIL.new("RGB", (312, 312))
    model.predict(_dummy, threshold=0.5)
    print("Warmup done")

    class_names = [""] + args.classes
    print(f"Classes: {args.classes}")

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else None

    if input_path.is_dir():
        images = [f for f in sorted(input_path.iterdir()) if f.suffix.lower() in IMAGE_EXTS]
        print(f"Running inference on {len(images)} images...")
        for img_path in images:
            run_on_image(model, img_path, output_dir, args.threshold, args.low_p, args.high_p, args.per_channel, class_names)
    else:
        run_on_image(model, input_path, output_dir, args.threshold, args.low_p, args.high_p, args.per_channel, class_names)


if __name__ == "__main__":
    main()
