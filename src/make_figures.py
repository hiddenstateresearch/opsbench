"""Generates the README figures from results/. Run from src/: python make_figures.py"""
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs"
OUT.mkdir(exist_ok=True)

LABELS = {
    "qwen3.6:35b": "Qwen3.6\n35B-A3B", "qwen2.5:7b": "Qwen2.5\n7B", "qwen2.5:14b": "Qwen2.5\n14B",
    "mistral-nemo:12b": "Mistral\nNemo 12B", "llama3.2:3b": "Llama3.2\n3B", "granite3.1-dense:8b": "Granite 3.1\nDense 8B",
    "llama3.2:1b": "Llama3.2\n1B", "command-r7b": "Command\nR7B",
}
BLUE, ORANGE, GRAY = "#4C72B0", "#DD8452", "#8c8c8c"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":"})


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return max(0, c - m), min(1, c + m)


data = json.loads((ROOT / "results/raw_results_full18_k3.json").read_text())
base = json.loads((ROOT / "results/baseline_results.json").read_text())

runs = defaultdict(lambda: defaultdict(list))
tokens = defaultdict(int)
for r in data:
    runs[r["model"]][r["scenario_id"]].append(r["score"]["passed"])
    u = r.get("usage", {})
    tokens[r["model"]] += u.get("input_tokens", 0) + u.get("output_tokens", 0)

stats = {}
for m, sc in runs.items():
    flags = [p for v in sc.values() for p in v]
    k, n = sum(flags), len(flags)
    lo, hi = wilson(k, n)
    stats[m] = dict(rate=k / n, p3=sum(all(v) for v in sc.values()) / len(sc), lo=lo, hi=hi)

base_flags = [r["score"]["passed"] for r in base]
base_rate = sum(base_flags) / len(base_flags)
order = sorted(stats, key=lambda m: -stats[m]["rate"])

# 1. pass rate vs pass^3
fig, ax = plt.subplots(figsize=(9, 4))
x = np.arange(len(order))
rate = np.array([stats[m]["rate"] * 100 for m in order])
err = np.clip(np.array([[stats[m]["rate"] - stats[m]["lo"] for m in order], [stats[m]["hi"] - stats[m]["rate"] for m in order]]) * 100, 0, None)
ax.bar(x - 0.2, rate, 0.4, yerr=err, capsize=3, color=BLUE, edgecolor="black", linewidth=0.5, label="Pass rate (95% CI)")
ax.bar(x + 0.2, [stats[m]["p3"] * 100 for m in order], 0.4, color=ORANGE, edgecolor="black", linewidth=0.5, label="pass$^3$")
ax.axhline(base_rate * 100, color=GRAY, linestyle="--", linewidth=1.2)
ax.text(len(order) - 0.45, base_rate * 100 + 1.5, f"scripted baseline ({base_rate:.0%})", ha="right", color="#555", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in order], fontsize=8)
ax.set_ylabel("%")
ax.set_ylim(0, 95)
ax.set_title("Passing once is not passing reliably", fontsize=11)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(OUT / "reliability_gap.png", dpi=160)
plt.close(fig)

# 2. token usage vs pass rate
fig, ax = plt.subplots(figsize=(6.5, 4))
offsets = {"qwen3.6:35b": (-10, 8), "qwen2.5:7b": (8, 4), "qwen2.5:14b": (8, -12), "mistral-nemo:12b": (8, 4),
           "llama3.2:3b": (8, 4), "granite3.1-dense:8b": (8, 4), "llama3.2:1b": (-8, 10), "command-r7b": (-78, -4)}
for m in order:
    ax.scatter(tokens[m] / 1000, stats[m]["rate"] * 100, s=45, color=BLUE, edgecolor="black", linewidth=0.6, zorder=3)
    ax.annotate(LABELS[m].replace("\n", " "), (tokens[m] / 1000, stats[m]["rate"] * 100),
                textcoords="offset points", xytext=offsets[m], fontsize=8)
a, b = "qwen2.5:14b", "qwen3.6:35b"
ax.plot([tokens[a] / 1000] * 2, [stats[a]["rate"] * 100, stats[b]["rate"] * 100], ":", color=ORANGE, linewidth=1.5, zorder=2)
ax.set_xlabel("Total tokens (thousands)")
ax.set_ylabel("Pass rate (%)")
ax.set_xlim(150, 1500)
ax.set_ylim(-8, 92)
ax.set_title("Equal token cost, 46 points apart", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "token_vs_accuracy.png", dpi=160)
plt.close(fig)

print("wrote", *sorted(p.name for p in OUT.glob("*.png")))
