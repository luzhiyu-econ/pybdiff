"""Render measured benchmark medians and full trial ranges."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager


def main(root, output, font=None):
    output.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(root / "summary.csv")
    if font:
        font_manager.fontManager.addfont(font)
        family = font_manager.FontProperties(fname=font).get_name()
    else:
        try:
            font_manager.findfont("Times New Roman", fallback_to_default=False)
            family = "Times New Roman"
        except ValueError:
            family = "DejaVu Serif"
    families = [family]
    plt.rcParams.update({"font.family": families + ["serif"], "font.size": 9,
                         "mathtext.fontset": "stix", "axes.unicode_minus": False,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": 0.6, "xtick.direction": "in",
                         "ytick.direction": "in", "savefig.dpi": 600})
    figure, axes = plt.subplots(1, 2, figsize=(7.3, 3.4), layout="constrained")
    for panel, case in zip(axes, data.case.unique()):
        rows = data[data.case == case].reset_index(drop=True)
        positions = np.arange(len(rows))
        colors = ["#666666" if engine == "Stata" else "#1f77b4" for engine in rows.engine]
        bars = panel.barh(positions, rows.median_seconds, color=colors, height=0.62,
                          edgecolor="black", linewidth=0.4)
        for bar, (_, row) in zip(bars, rows.iterrows()):
            if row.engine == "Stata":
                bar.set_hatch("///")
            panel.text(row.max_seconds + rows.max_seconds.max() * .02,
                       bar.get_y() + bar.get_height() / 2, f"{row.median_seconds:.2f}s",
                       va="center", fontsize=8)
        error = np.vstack([rows.median_seconds - rows.min_seconds,
                           rows.max_seconds - rows.median_seconds])
        panel.errorbar(rows.median_seconds, positions, xerr=error, fmt="none",
                       ecolor="black", capsize=2, linewidth=.65)
        labels = []
        for _, row in rows.iterrows():
            unit = "core" if row.engine == "Stata" else "worker"
            suffix = "s" if row.workers != 1 else ""
            labels.append(f"{row.engine} ({int(row.workers)} {unit}{suffix})")
        panel.set_yticks(positions, labels)
        panel.invert_yaxis()
        panel.set_xlabel("Elapsed time (seconds)")
        panel.set_xlim(0, rows.max_seconds.max() * 1.3)
        panel.text(0, 1.03, f"({'a' if case == 'OLS' else 'b'}) {case}",
                   transform=panel.transAxes, fontsize=10)
        panel.grid(axis="x", alpha=.15)
        panel.set_axisbelow(True)
    figure.savefig(output / "runtime.png", bbox_inches="tight", pad_inches=.05)
    figure.savefig(output / "runtime.pdf", bbox_inches="tight", pad_inches=.05)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("docs/benchmarks"))
    parser.add_argument("--output", type=Path, default=Path(".local/figures"))
    parser.add_argument("--font", type=Path, help="Optional local Times New Roman TTF file")
    args = parser.parse_args()
    main(args.input, args.output, args.font)
