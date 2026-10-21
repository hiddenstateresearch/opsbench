"""Reproduces every statistic reported in the paper (Wilson CIs, Spearman
correlations) directly from the released raw results.
Run from src/: python paper_stats.py
"""
import json
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0, center - margin), min(1, center + margin))


def main():
    data = json.loads((RESULTS_DIR / "raw_results_full18_k3.json").read_text())
    by_scenario = defaultdict(lambda: defaultdict(list))
    tokens_by_model = defaultdict(int)
    for r in data:
        by_scenario[r["model"]][r["scenario_id"]].append(r["score"]["passed"])
        u = r.get("usage", {})
        tokens_by_model[r["model"]] += u.get("input_tokens", 0) + u.get("output_tokens", 0)

    results = {}
    for model, scenarios in by_scenario.items():
        flags = [p for runs in scenarios.values() for p in runs]
        k, n = sum(flags), len(flags)
        pass3 = sum(1 for runs in scenarios.values() if all(runs)) / len(scenarios)
        results[model] = {"pass_rate": k / n, "pass3": pass3, "n": n, "k": k}

    print("=== Table 2: pass rate, pass^3, and 95% Wilson CI on pass rate ===")
    for model, r in results.items():
        lo, hi = wilson_ci(r["k"], r["n"])
        print(f"{model}: pass_rate={r['pass_rate']*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}] "
              f"pass3={r['pass3']*100:.1f}% tokens={tokens_by_model[model]:,}")

    print("\n=== Spearman: total parameter count vs. pass rate ===")
    params = {
        "qwen3.6:35b": 35, "qwen2.5:14b": 14, "qwen2.5:7b": 7, "mistral-nemo:12b": 12,
        "llama3.2:3b": 3, "llama3.2:1b": 1, "command-r7b": 7, "granite3.1-dense:8b": 8,
    }
    models = list(results.keys())
    xs = [params[m] for m in models]
    ys = [results[m]["pass_rate"] for m in models]
    rho, p = spearmanr(xs, ys)
    print(f"n=8 (all models): rho={rho:.4f} p={p:.4f}")

    excl = [m for m in models if m not in ("command-r7b", "granite3.1-dense:8b")]
    xs2 = [params[m] for m in excl]
    ys2 = [results[m]["pass_rate"] for m in excl]
    rho2, p2 = spearmanr(xs2, ys2)
    print(f"n=6 (excluding Command-R7B/Granite, non-capability zeros): rho={rho2:.4f} p={p2:.4f}")

    rho3, p3 = spearmanr([tokens_by_model[m] for m in models], ys)
    print(f"total tokens vs. pass rate, n=8: rho={rho3:.4f} p={p3:.4f}")

    print("\n=== Qwen 2.5 14B vs. Qwen3.6-35B-A3B: token/wall-clock comparison ===")
    wc_by_model = defaultdict(float)
    for r in data:
        wc_by_model[r["model"]] += r["wall_clock_s"]
    t14, t36 = tokens_by_model["qwen2.5:14b"], tokens_by_model["qwen3.6:35b"]
    wc14, wc36 = wc_by_model["qwen2.5:14b"], wc_by_model["qwen3.6:35b"]
    print(f"token gap: {abs(t14-t36)/max(t14,t36)*100:.2f}% apart ({t14:,} vs {t36:,})")
    print(f"pass rate gap: {abs(results['qwen2.5:14b']['pass_rate']-results['qwen3.6:35b']['pass_rate'])*100:.1f} points")
    print(f"wall-clock: {(wc36/wc14-1)*100:.1f}% longer for Qwen3.6-35B-A3B ({wc36:.1f}s vs {wc14:.1f}s)")

    print("\n=== Scripted baseline ===")
    base = json.loads((RESULTS_DIR / "baseline_results.json").read_text())
    base_runs = defaultdict(list)
    for r in base:
        base_runs[r["scenario_id"]].append(r["score"]["passed"])
    flags = [p for runs in base_runs.values() for p in runs]
    print(f"pass rate={sum(flags)/len(flags)*100:.1f}% ({sum(flags)}/{len(flags)}) "
          f"pass3={sum(all(r) for r in base_runs.values())}/{len(base_runs)}")
    failed_all = sorted(s for s, runs in base_runs.items() if not any(runs))
    qwen_hits = sum(any(by_scenario["qwen3.6:35b"][s]) for s in failed_all)
    print(f"baseline fails every run of {len(failed_all)} scenarios; Qwen3.6-35B-A3B passes {qwen_hits} of them at least once")

    print("\n=== Granite 3.1 Dense: without vs. with plain-text tool-call parsing ===")
    raw_only = json.loads((RESULTS_DIR / "granite_without_text_call_parsing.json").read_text())
    print(f"Ollama-native parsing only: {sum(r['score']['passed'] for r in raw_only)}/{len(raw_only)} passed")
    with_parse = [r["score"]["passed"] for r in data if r["model"] == "granite3.1-dense:8b"]
    print(f"with harness text parsing (Table 2): {sum(with_parse)}/{len(with_parse)} passed")

    print("\n=== Granite 3.1 Dense: dispatched tool calls ===")
    is_err = lambda e: isinstance(e["result"], dict) and "error" in e["result"]
    errs = missing = recovered = calls = 0
    for r in data:
        if r["model"] != "granite3.1-dense:8b":
            continue
        log = r["tool_log"]
        calls += len(log)
        for i, e in enumerate(log):
            if not is_err(e):
                continue
            errs += 1
            msg = str(e["result"]["error"])
            # Two shapes: a TypeError ("missing 1 required positional argument") from
            # tools taking **kwargs, or a bare KeyError naming the absent field.
            key = msg.strip("'\"")
            if ("missing" in msg and "argument" in msg) or (key.isidentifier() and key not in e["input"]):
                missing += 1
            if any(x["tool"] == e["tool"] and not is_err(x) for x in log[i + 1:]):
                recovered += 1
    print(f"dispatched={calls} errored={errs} missing_required_arg={missing} later_same_tool_success={recovered}")

    print("\n=== Alternate user-simulator check (phi4:14b) ===")
    alt = json.loads((RESULTS_DIR / "altsim_check.json").read_text())
    pairs = {(r["model"], r["scenario_id"]) for r in alt}
    orig = [r["score"]["passed"] for r in data if (r["model"], r["scenario_id"]) in pairs]
    altp = [r["score"]["passed"] for r in alt]
    early = sum("[END_CHAT]" in r["transcript"][0]["content"] for r in alt)
    print(f"llama3.1:8b sim: {sum(orig)}/{len(orig)}  phi4:14b sim: {sum(altp)}/{len(altp)}  "
          f"phi4 ends chat in opening message: {early}/{len(alt)}")


if __name__ == "__main__":
    main()
