#!/usr/bin/env python3
"""
Reference inference for models produced by gen.py (Conv -> Relu -> Add with broadcast bias).
Uses PyTorch to mirror the ONNX graph; writes float32 little-endian golden.bin for kernel_driver --compare.

Usage:
  python3 pytorch_reference.py [--onnx path] [--out golden.bin] [--sum-only]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper


def load_conv_and_bias(onnx_path: Path) -> tuple[np.ndarray, np.ndarray]:
    model = onnx.load(onnx_path)
    w = b = None
    for init in model.graph.initializer:
        if init.name == "W":
            w = numpy_helper.to_array(init)
        elif init.name == "add_bias":
            b = numpy_helper.to_array(init)
    if w is None or b is None:
        raise RuntimeError("Expected initializers W and add_bias in ONNX")
    return w.astype(np.float32), b.astype(np.float32)


class RefNet(nn.Module):
    """Matches end2end/gen.py: Conv 3x3, stride 1, pad 0; ReLU; Add bias [1,1,3,3]."""

    def __init__(self, w: np.ndarray, bias: np.ndarray) -> None:
        super().__init__()
        # w: [out_c, in_c, kh, kw]
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.conv.weight.data = torch.from_numpy(w.copy())
        self.register_buffer("add_bias", torch.from_numpy(bias.copy()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        y = torch.relu(y)
        return y + self.add_bias


def main() -> None:
    ap = argparse.ArgumentParser(description="PyTorch reference for test_model.onnx")
    ap.add_argument(
        "--onnx",
        type=Path,
        default=Path(__file__).parent / "models" / "test_model.onnx",
        help="Path to ONNX model",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "golden.bin",
        help="Output float32 binary (9 elements for 1x1x3x3)",
    )
    ap.add_argument("--sum-only", action="store_true", help="Print sum only, do not write file")
    args = ap.parse_args()

    w, bias = load_conv_and_bias(args.onnx)
    net = RefNet(w, bias)
    net.eval()

    x = torch.zeros(1, 1, 5, 5, dtype=torch.float32)
    x[0, 0, 0, 0] = 1.0

    with torch.no_grad():
        y = net(x).numpy().astype(np.float32).ravel()

    s = float(y.sum())
    print(f"pytorch_reference: output sum = {s:g}")
    if args.sum_only:
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(y.tobytes())
    print(f"Wrote {len(y)} floats to {args.out}")


if __name__ == "__main__":
    main()
