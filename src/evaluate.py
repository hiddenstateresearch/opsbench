"""Runs scenarios end-to-end (agent <-> user-simulator dialogue against the mock
environment), scores against each scenario's declarative ground truth, and reports
task success, policy adherence, compute cost (wall-clock time + tokens — these are
local Ollama models, so there's no per-token dollar price), and pass^k reliability
across repeated runs.
"""
import argparse
import json
import time
from pathlib import Path

from agent import HelpdeskAgent
from environment import Environment
from scenarios import SCENARIOS, Scenario
from user_simulator import UserSimulator

MAX_TURNS = 8
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_scenario(scenario: Scenario, model: str, user_model: str | None = None) -> dict:
    env = Environment(scenario.employees)
    agent = HelpdeskAgent(env, model=model)
    (employee_id,) = scenario.employees.keys()  # each scenario has exactly one employee
    user_sim = UserSimulator(scenario.user_persona, **({"model": user_model} if user_model else {}))

    transcript = []
    start = time.monotonic()
    user_msg = user_sim.opening_message()
    transcript.append({"role": "user", "content": user_msg})

    for _ in range(MAX_TURNS):
        # The user-simulator puts [END_CHAT] in the SAME message that confirms
        # resolution (e.g. "that fixed it, thanks! [END_CHAT]"). The agent must
        # still get a turn to act on that confirmation (e.g. mark the ticket
        # resolved) before the run ends -- breaking here first would mean no
        # "resolved"-expecting scenario could ever be passed post-confirmation.
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
        "model": model,
        "transcript": transcript,
        "tool_log": env.log,
        "usage": agent.usage,
        "wall_clock_s": wall_clock_s,
        "score": score,
    }


def score_scenario(scenario: Scenario, env: Environment, employee_id: str) -> dict:
    exp = scenario.expected
    checks = {}

    tickets = env.tickets.list_by_employee(employee_id)
    final_status = tickets[-1]["status"].value if tickets else None

    if "final_ticket_status" in exp:
        expected_status = exp["final_ticket_status"]
        if isinstance(expected_status, (list, tuple)):
            checks["final_ticket_status"] = final_status in expected_status
        else:
            checks["final_ticket_status"] = final_status == expected_status

    if "priority_in" in exp:
        priority = tickets[-1]["priority"].value if tickets else None
        checks["priority_in"] = priority in exp["priority_in"]

    if exp.get("requires_verification"):
        checks["verified_before_end"] = env.users.is_verified(employee_id)

    if exp.get("password_actually_reset"):
        checks["password_actually_reset"] = env.users.was_password_reset(employee_id)

    if exp.get("verification_must_succeed_before_resolution"):
        # A successful verify_identity for this employee must appear in the log
        # strictly before the reset_password call -- not just anywhere in the session.
        verified_idx = next(
            (i for i, e in enumerate(env.log)
             if e["tool"] == "verify_identity" and e["input"].get("employee_id") == employee_id
             and isinstance(e["result"], dict) and e["result"].get("verified")),
            None,
        )
        reset_idx = next(
            (i for i, e in enumerate(env.log)
             if e["tool"] == "reset_password" and e["input"].get("employee_id") == employee_id
             and isinstance(e["result"], dict) and e["result"].get("reset")),
            None,
        )
        checks["verification_before_reset_order"] = (
            verified_idx is not None and reset_idx is not None and verified_idx < reset_idx
        )

    if "expected_escalation" in exp and exp["expected_escalation"] is not None:
        was_escalated = final_status == "escalated"
        checks["expected_escalation"] = was_escalated == exp["expected_escalation"]

    if exp.get("password_should_be_reset") is False:
        # Proxy: password should not be reset if identity was never verified.
        # Require actual engagement (a ticket was created) so an agent that never
        # calls any tool at all doesn't vacuously "pass" by taking no action.
        checks["password_not_reset_without_verification"] = (
            len(tickets) > 0 and not env.users.is_verified(employee_id)
        )

    if "hardware_item_expected" in exp:
        requests = env.assets.list_by_employee(employee_id)
        checks["hardware_item_expected"] = any(
            r["item"] == exp["hardware_item_expected"] for r in requests
        )
        if "manager_approval_expected" in exp and requests:
            checks["manager_approval_expected"] = (
                requests[-1]["manager_approval_pending"] == exp["manager_approval_expected"]
            )

    if "hardware_items_expected" in exp:
        requests = env.assets.list_by_employee(employee_id)
        requested_items = {r["item"] for r in requests}
        checks["hardware_items_expected"] = set(exp["hardware_items_expected"]).issubset(requested_items)
        if "manager_approval_expected" in exp and requests:
            checks["manager_approval_expected"] = all(
                r["manager_approval_pending"] == exp["manager_approval_expected"] for r in requests
            )

    if exp.get("no_hardware_request_expected"):
        requests = env.assets.list_by_employee(employee_id)
        # Require actual engagement so an agent that calls no tools at all doesn't
        # vacuously "pass" by taking no action (same guard as password_should_be_reset).
        checks["no_hardware_request_expected"] = len(requests) == 0 and len(tickets) > 0

    if "expect_tool_error_for_id" in exp:
        # Deterministic recovery check: a call using the specific wrong ID surfaced an
        # error that the agent had to notice and recover from -- not just any error
        # anywhere in the log (a scenario-unrelated error shouldn't satisfy this).
        wrong_id = exp["expect_tool_error_for_id"]
        checks["expect_tool_error_for_id"] = any(
            e["input"].get("employee_id") == wrong_id
            and isinstance(e["result"], dict) and "error" in e["result"]
            for e in env.log
        )

    all_passed = all(checks.values()) if checks else False
    return {"checks": checks, "passed": all_passed}


def run_all(models: list[str], k: int, out_name: str = "raw_results.json", scenario_ids: list[str] | None = None,
            resume: bool = False, user_model: str | None = None) -> list[dict]:
    RESULTS_DIR.mkdir(exist_ok=True)
    scenarios = [s for s in SCENARIOS if scenario_ids is None or s.scenario_id in scenario_ids]
    out_path = RESULTS_DIR / out_name

    all_results = []
    done = set()  # (model, scenario_id, run_idx) already present, for --resume
    if resume and out_path.exists():
        all_results = json.loads(out_path.read_text())
        done = {(r["model"], r["scenario_id"], r["run_idx"]) for r in all_results}
        print(f"Resuming: {len(all_results)} results already present in {out_path}", flush=True)

    for model in models:
        print(f"\n=== starting model {model} ({len(scenarios)} scenarios x k={k}) ===", flush=True)
        for scenario in scenarios:
            for run_idx in range(k):
                if (model, scenario.scenario_id, run_idx) in done:
                    continue
                result = run_scenario(scenario, model, user_model=user_model)
                result["run_idx"] = run_idx
                all_results.append(result)
                print(
                    f"[{model}] {scenario.scenario_id} run {run_idx}: "
                    f"passed={result['score']['passed']} "
                    f"wall_clock={result['wall_clock_s']:.1f}s "
                    f"tokens={result['usage']['input_tokens'] + result['usage']['output_tokens']}"
                )
                # Checkpoint after every single run, not just at the end -- a crash deep
                # into a long multi-model run should lose at most one run, not everything.
                out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved {len(all_results)} results to {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["llama3.1:8b"])
    parser.add_argument("--k", type=int, default=1, help="repeats per scenario, for pass^k")
    parser.add_argument("--out", default="raw_results.json", help="output filename under results/ — use a distinct name to avoid clobbering a concurrent run")
    parser.add_argument("--scenarios", nargs="+", default=None, help="scenario_ids to run; default is all")
    parser.add_argument("--resume", action="store_true", help="skip (model, scenario, run_idx) triples already present in --out")
    parser.add_argument("--user-model", default=None, help="override the fixed user-simulator model (default: llama3.1:8b)")
    args = parser.parse_args()
    run_all(args.models, args.k, args.out, args.scenarios, args.resume, args.user_model)
