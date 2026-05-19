#!/usr/bin/env python3
"""Export an RF-DETR segmentation model to ONNX (default) or TorchScript.

ONNX is produced via rfdetr's built-in exporter. TorchScript is produced by
tracing the inference-optimized model and saved as a standalone `.ts` file
that can be loaded with only `torch` installed (no `rfdetr`) on a target
device such as a Jetson, via `torch.jit.load(path)`.
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from rfdetr import RFDETRSegNano, RFDETRSegSmall
from rfdetr.models.heads.segmentation import DepthwiseConvBlock


_SIZE_TO_CLASS = {"nano": RFDETRSegNano, "small": RFDETRSegSmall}


def _patch_depthwise_conv_for_export():
    """Replace the custom autograd.Function with plain F.conv2d.

    The original wraps conv2d in a torch.autograd.Function to disable cuDNN
    in the backward pass (T4/P100 workaround). That Function is not
    serializable by torch.jit.save, and we don't need it for inference.
    """
    def _depthwise_conv(self, x):
        return F.conv2d(
            x, self.dwconv.weight, self.dwconv.bias,
            self.dwconv.stride, self.dwconv.padding,
            self.dwconv.dilation, self.dwconv.groups,
        )
    DepthwiseConvBlock._depthwise_conv = _depthwise_conv


def load_model(model_path: str, num_classes: int):
    name = Path(model_path).stem.lower()
    cls = next((v for k, v in _SIZE_TO_CLASS.items() if k in name), None)
    if cls is None:
        raise ValueError(f"Cannot infer model size from '{model_path}'. Expected 'nano' or 'small' in filename.")
    return cls(pretrain_weights=model_path, num_classes=num_classes, device="cuda")


def export_onnx(model, args):
    output_dir = args.output if args.output else "output"
    print(f"Exporting ONNX to {output_dir}/ (opset {args.opset}, batch {args.batch_size}, dynamic_batch={args.dynamic_batch})")
    model.export(
        output_dir=output_dir,
        opset_version=args.opset,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
    )


def export_torchscript(model, model_path: str, args):
    _patch_depthwise_conv_for_export()
    resolution = model.model.resolution
    print(f"Tracing at resolution {resolution}x{resolution}, batch {args.batch_size}, dtype {args.dtype}")

    model.optimize_for_inference(compile=True, batch_size=args.batch_size, dtype=args.dtype)
    ts_module = model.model.inference_model
    assert isinstance(ts_module, torch.jit.ScriptModule), "expected a traced ScriptModule"

    out = Path(args.output) if args.output else Path(model_path).with_suffix(".ts")
    torch.jit.save(ts_module, str(out))
    print(f"Saved TorchScript module to {out}")
    print(f"Expected input: tensor [{args.batch_size}, 3, {resolution}, {resolution}] dtype={args.dtype} on cuda")
    print("Outputs: (boxes_cxcywh_normalized [B,Q,4], class_logits [B,Q,C], masks [B,Q,H,W])")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model", nargs="?",
                   default="/home/ratul/Workstation/hexafarms/leo_pipe_detection/rosbags/saved_models/pipe_seg_nano.pt",
                   help="Path to .pt model file")
    p.add_argument("--format", choices=["onnx", "torchscript"], default="onnx",
                   help="Export format (default: onnx)")
    p.add_argument("-o", "--output", default=None,
                   help="Output path: directory for onnx (default: 'output/'), file for torchscript (default: <model>.ts)")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                   help="TorchScript only")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    p.add_argument("--dynamic-batch", action="store_true", help="ONNX: export with dynamic batch dimension")
    args = p.parse_args()

    model = load_model(args.model, args.num_classes)

    if args.format == "onnx":
        export_onnx(model, args)
    else:
        export_torchscript(model, args.model, args)


if __name__ == "__main__":
    main()
