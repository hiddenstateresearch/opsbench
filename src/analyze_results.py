"""Aggregate raw_results.json into pass^k reliability and cost/accuracy summary per model.

'Cost' here is compute cost (wall-clock time + tokens processed), not API dollars —
these are local Ollama models. This is actually the more honest cost signal for
comparing local-model deployment options than an opaque per-token API price.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_results(filename: str = "raw_results.json") -> list[dict]:
    return json.loads((RESULTS_DIR / filename).read_text())


def summarize(results: list[dict]) -> dict:
    by_model_scenario = defaultdict(list)
    for r in results:
        by_model_scenario[(r["model"], r["scenario_id"])].append(r)

    per_model = defaultdict(lambda: {"scenarios": {}, "total_wall_clock_s": 0.0, "total_tokens": 0, "n_runs": 0})

    for (model, scenario_id), runs in by_model_scenario.items():
        passed_flags = [r["score"]["passed"] for r in runs]
        pass_at_k = all(passed_flags)  # pass^k: succeeds on every run, per tau-bench convention
        pass_rate = sum(passed_flags) / len(passed_flags)
        avg_wall_clock = sum(r["wall_clock_s"] for r in runs) / len(runs)
        avg_tokens = sum(r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in runs) / len(runs)

        per_model[model]["scenarios"][scenario_id] = {
            "pass_rate": pass_rate,
            "pass_at_k": pass_at_k,
            "avg_wall_clock_s": avg_wall_clock,
            "avg_tokens": avg_tokens,
            "n_runs": len(runs),
        }
        per_model[model]["total_wall_clock_s"] += sum(r["wall_clock_s"] for r in runs)
        per_model[model]["total_tokens"] += sum(
            r["usage"]["input_tokens"] + r["usage"]["output_tokens"] for r in runs
        )
        per_model[model]["n_runs"] += len(runs)

    for model, data in per_model.items():
        rates = [s["pass_rate"] for s in data["scenarios"].values()]
        pass_at_k_rates = [s["pass_at_k"] for s in data["scenarios"].values()]
        data["overall_pass_rate"] = sum(rates) / len(rates)
        data["overall_pass_at_k"] = sum(pass_at_k_rates) / len(pass_at_k_rates)

    return dict(per_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_files", nargs="+", default=["raw_results.json"])
    args = parser.parse_args()
    results = []
    for f in args.in_files:
        results.extend(load_results(f))
    summary = summarize(results)
    print(json.dumps(summary, indent=2))

    print("\n--- Compute-cost / Accuracy Pareto summary ---")
    for model, data in summary.items():
        print(
            f"{model}: overall_pass_rate={data['overall_pass_rate']:.2f} "
            f"pass^k={data['overall_pass_at_k']:.2f} "
            f"total_wall_clock={data['total_wall_clock_s']:.1f}s "
            f"total_tokens={data['total_tokens']} (n={data['n_runs']})"
        )
