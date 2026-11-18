import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dataset import CLASSES, DISEASE_IDX

runs_dir = sys.argv[1] if len(sys.argv) > 1 else "/kaggle/working/runs"
by_arch = {}
for path in sorted(glob.glob(os.path.join(runs_dir, "*", "results.json"))):
    with open(path) as f:
        r = json.load(f)
    by_arch.setdefault(r["arch"], []).append(r)

if not by_arch:
    sys.exit(f"No results.json files found under {runs_dir}")


def mean_std(values):
    v = np.array(values, dtype=float)
    return float(np.nanmean(v)), (float(np.nanstd(v, ddof=1)) if len(v) > 1 else 0.0)


archs = sorted(by_arch)
lines = []

header = "| Class | " + " | ".join(f"{a} (n={len(by_arch[a])})" for a in archs) + " |"
lines += [header, "|" + "---|" * (len(archs) + 1)]
stats = {a: {} for a in archs}
for c in CLASSES:
    row = f"| {c} |"
    for a in archs:
        m, s = mean_std([r["test_auc_per_class"][c] for r in by_arch[a]])
        stats[a][c] = (m, s)
        row += f" {m:.3f} ± {s:.3f} |"
    lines.append(row)
row = "| **Mean (14 diseases)** |"
for a in archs:
    m, s = mean_std([r["test_mean_auc"] for r in by_arch[a]])
    row += f" **{m:.3f} ± {s:.3f}** |"
lines.append(row)

lines += ["", "| Metric | " + " | ".join(archs) + " |", "|" + "---|" * (len(archs) + 1)]


def cost_row(label, fn, fmt):
    cells = []
    for a in archs:
        vals = [fn(r) for r in by_arch[a] if fn(r) is not None]
        cells.append(fmt.format(np.mean(vals)) if vals else "n/a")
    return f"| {label} | " + " | ".join(cells) + " |"


lines.append(cost_row("Parameters (millions)", lambda r: r["n_params"] / 1e6, "{:.1f}"))
lines.append(cost_row("Seconds per epoch", lambda r: float(np.mean([h["seconds"] for h in r["history"]])), "{:.0f}"))
lines.append(cost_row("Peak GPU memory (GB)", lambda r: r["peak_gpu_mem_gb"], "{:.1f}"))
lines.append(cost_row("Best epoch", lambda r: r["best_epoch"], "{:.1f}"))

text = "\n".join(lines)
print(text)
with open(os.path.join(runs_dir, "comparison.md"), "w") as f:
    f.write(text + "\n")

disease = [CLASSES[i] for i in DISEASE_IDX]
x = np.arange(len(disease))
w = 0.8 / len(archs)
plt.figure(figsize=(11, 5))
for k, a in enumerate(archs):
    means = [stats[a][c][0] for c in disease]
    stds = [stats[a][c][1] for c in disease]
    plt.bar(x + k * w, means, w, yerr=stds, capsize=2, label=a)
plt.xticks(x + w * (len(archs) - 1) / 2, disease, rotation=45, ha="right")
plt.ylabel("Test AUC")
plt.ylim(0.5, 1.0)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(runs_dir, "per_class_auc.png"), dpi=150)
print(f"\nSaved comparison.md and per_class_auc.png to {runs_dir}")
