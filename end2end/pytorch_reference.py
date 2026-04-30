import argparse
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

from gen import ONNX_MODEL_PATH


def load_initializers(onnx_path: Path) -> dict[str, np.ndarray]:
    model = onnx.load(onnx_path)
    return {init.name: numpy_helper.to_array(init).astype(np.float32) for init in model.graph.initializer}


class LeNetRef(nn.Module):
    def __init__(self, w: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2, bias=True)
        self.conv1.weight.data = torch.from_numpy(w["W0"].copy())
        self.conv1.bias.data = torch.from_numpy(w["B0"].copy())

        self.s2 = nn.Conv2d(6, 6, kernel_size=2, stride=2, bias=True)
        self.s2.weight.data = torch.from_numpy(w["W_s2"].copy())
        self.s2.bias.data = torch.from_numpy(w["B_s2"].copy())

        self.c3 = nn.Conv2d(6, 16, kernel_size=5, bias=True)
        self.c3.weight.data = torch.from_numpy(w["W_c3"].copy())
        self.c3.bias.data = torch.from_numpy(w["B_c3"].copy())

        self.s4 = nn.Conv2d(16, 16, kernel_size=2, stride=2, bias=True)
        self.s4.weight.data = torch.from_numpy(w["W_s4"].copy())
        self.s4.bias.data = torch.from_numpy(w["B_s4"].copy())

        self.c5 = nn.Conv2d(16, 120, kernel_size=5, bias=True)
        self.c5.weight.data = torch.from_numpy(w["W_c5"].copy())
        self.c5.bias.data = torch.from_numpy(w["B_c5"].copy())

        self.f6 = nn.Conv2d(120, 84, kernel_size=1, bias=True)
        self.f6.weight.data = torch.from_numpy(w["W_f6"].copy())
        self.f6.bias.data = torch.from_numpy(w["B_f6"].copy())

        self.out = nn.Conv2d(84, 10, kernel_size=1, bias=True)
        self.out.weight.data = torch.from_numpy(w["W_out"].copy())
        self.out.bias.data = torch.from_numpy(w["B_out"].copy())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.s2(x))
        x = torch.relu(self.c3(x))
        x = torch.relu(self.s4(x))
        x = torch.relu(self.c5(x))
        x = torch.relu(self.f6(x))
        x = self.out(x)
        return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", type=Path, default=ONNX_MODEL_PATH)
    args = ap.parse_args()

    weights = load_initializers(args.onnx)
    net = LeNetRef(weights)
    net.eval()

    x = torch.zeros(1, 1, 28, 28, dtype=torch.float32)
    for i in range(28):
        for j in range(28):
            x[0, 0, i, j] = (i * 28 + j) / 783.0

    with torch.no_grad():
        y = net(x).numpy()

    print("PyTorch output shape:", y.shape)
    print("PyTorch output:\n", y.reshape(10))
    print("Class index:", int(np.argmax(y.reshape(10))))

if __name__ == "__main__":
    main()
