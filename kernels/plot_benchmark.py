#!/usr/bin/env python3

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[dict[str, str]]:
    lines = [
        ln
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    return list(csv.DictReader(lines))


def split_name(raw: str) -> tuple[str, tuple[int, ...]]:
    s = raw.strip().strip('"').split("/")
    return s[0], tuple(int(x) for x in s[1:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("kernels_bench.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("kernels/figures"))
    ap.add_argument("--log", action="store_true")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    if not rows:
        raise SystemExit(f"No data found: {args.csv}")

    gemm: dict[str, list[tuple[int, float]]] = defaultdict(list)
    conv: dict[str, list[tuple[tuple[int, int], float]]] = defaultdict(list)

    for row in rows:
        base, argv = split_name(row["name"])
        t = float(row["cpu_time"])
        if base.startswith("bench_matmul_") and len(argv) == 1:
            gemm[base.removeprefix("bench_matmul_")].append((argv[0], t))
        elif base.startswith("bench_conv_") and len(argv) == 2:
            conv[base.removeprefix("bench_conv_")].append((argv, t))

    for pts in gemm.values():
        pts.sort(key=lambda x: x[0])
    for pts in conv.values():
        pts.sort(key=lambda x: x[0])

    import matplotlib.pyplot as plt

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def style(ax, title: str) -> None:
        ax.set_title(title)
        ax.set_ylabel("cpu_time, ns")
        ax.grid(True, alpha=0.35)
        if args.log:
            ax.set_yscale("log")
        ax.legend()

    if gemm:
        _, ax = plt.subplots(figsize=(8, 5))
        for name in sorted(gemm):
            xs, ys = zip(*gemm[name])
            ax.plot(xs, ys, marker="o", label=name)
        ax.set_xlabel(r"$N$ ($N \times N \times N$ GEMM)")
        style(ax, "GEMM")
        plt.tight_layout()
        plt.savefig(out / "gemm_bench.png", dpi=150)
        plt.close()
        print(out / "gemm_bench.png")

    if conv:
        keys = sorted({k for pts in conv.values() for k, _ in pts})
        xs = range(len(keys))
        lbl = [rf"$C_{{in}}$={a}, $H$={b}" for a, b in keys]

        _, ax = plt.subplots(figsize=(8, 5))
        for name in sorted(conv):
            d = {k: t for k, t in conv[name]}
            ax.plot(
                xs,
                [d.get(k, float("nan")) for k in keys],
                marker="o",
                label=name,
            )
        ax.set_xticks(list(xs), lbl, rotation=15, ha="right")
        style(ax, r"Conv2d valid, s=1, p=0 ($C_{out}=8$, $K=3$)")
        plt.tight_layout()
        plt.savefig(out / "conv_bench.png", dpi=150)
        plt.close()
        print(out / "conv_bench.png")


if __name__ == "__main__":
    main()