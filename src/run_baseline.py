"""Runs the trivial scripted baseline (no LLM agent) through the same dialogue
loop, environment, and scoring as evaluate.py, for a pass-rate floor to compare
real agent models against. User-simulator is unchanged (still a sampled LLM),
so pass^k here still reflects simulator variance, same caveat as Section 3.

Usage: python run_baseline.py --k 3 --out baseline_results.json
"""
import argparse
import json
import time
from pathlib import Path

from baseline_agent import ScriptedBaselineAgent
from environment import Environment
from evaluate import score_scenario, MAX_TURNS
from scenarios import SCENARIOS
from user_simulator import UserSimulator

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_scenario_baseline(scenario) -> dict:
    env = Environment(scenario.employees)
    agent = ScriptedBaselineAgent(env)
    (employee_id,) = scenario.employees.keys()
    user_sim = UserSimulator(scenario.user_persona)

    transcript = []
    start = time.monotonic()
    user_msg = user_sim.opening_message()
    transcript.append({"role": "user", "content": user_msg})

    for _ in range(MAX_TURNS):
        done = UserSimulator.is_done(user_msg)
        agent_reply = agent.respond_to_user(user_msg)
        transcript.append({"role": "agent", "content": agent_reply})
        if done:
            break
        user_msg = user_sim.reply_to(agent_reply)
        transcript.append({"role": "user", "content": user_msg})
    wall_clock_s = time.monotonic() - start

    score = score_scenario(scenario, env, employee_id)

    return {
        "scenario_id": scenario.scenario_id,
        "model": "scripted-baseline",
        "transcript": transcript,
        "tool_log": env.log,
        "usage": agent.usage,
        "wall_clock_s": wall_clock_s,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--out", default="baseline_results.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / args.out
    all_results = []
    for scenario in SCENARIOS:
        for run_idx in range(args.k):
            result = run_scenario_baseline(scenario)
            result["run_idx"] = run_idx
            all_results.append(result)
            print(f"{scenario.scenario_id} run {run_idx}: passed={result['score']['passed']}")
            out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved {len(all_results)} results to {out_path}")


if __name__ == "__main__":
    main()
