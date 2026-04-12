import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

from gen import ONNX_MODEL_PATH


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
        default=ONNX_MODEL_PATH,
        help="Path to ONNX model",
    )
    ap.add_argument("--sum-only", action="store_true", help="Print sum only, do not write file")
    args = ap.parse_args()

    w, bias = load_conv_and_bias(args.onnx)
    net = RefNet(w, bias)
    net.eval()

    x = torch.zeros(1, 1, 5, 5, dtype=torch.float32)
    x[0, 0, 0, 0] = 1.0

    with torch.no_grad():
        y = net(x).numpy()
        
    print(f"Pytorch output:\n{y}")
    if args.sum_only:
        return


if __name__ == "__main__":
    main()
