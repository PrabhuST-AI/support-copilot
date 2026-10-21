"""react_agent.py -- ReAct loop implementation with safety limits, graceful fallbacks, and resumable execution.

Tasks:
1. Implement run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10.0):
   - Messages start as a system message containing the word AGENT and the user message.
   - Enforce three limits and graceful fallbacks:
     (a) After max_iterations model calls without a final answer: stop with status FALLBACK_MAX_ITERATIONS (S7 ends this way).
     (b) After each model call add usage.total_tokens to tokens_used; if it exceeds token_limit, stop immediately
         (without executing the tool just requested) with status FALLBACK_TOKEN_LIMIT.
     (c) At the top of each iteration, if elapsed >= time_limit_seconds (measured via time.monotonic()),
         stop with status FALLBACK_TIMEOUT (stops before first model call when time_limit_seconds=0).
   - After every tool observation is appended, write a checkpoint file in state/ named after run_id:
     contains run_id, scenario_id, messages, tokens_used, step_number, tools_executed, etc.
   - Every fallback returns the steps completed so far and a short human-handoff summary string in final_answer.
   - Return dictionary: {"scenario_id", "status", "final_answer", "steps", "tools_executed", "llm_calls", "tokens_used", "refund_amount"}.
2. Implement resume_agent(run_id) that loads the checkpoint from state/<run_id>.json and continues execution seamlessly.
3. Demonstrate S1 crash recovery:
   - Raise simulated crash right after 2nd tool execution (after checkpoint is saved).
   - Call resume_agent(run_id).
   - Tools executed across both parts must be exactly: get_order, search_policy, calculate_refund, issue_refund (4 executions, none repeated).
   - Final status must be REFUND with 128.25.
4. Record all results in README.md Part D.
"""

import csv
import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from agent_tools import TOOL_MAP, TOOLS_SPEC
from mock_llm import chat, reset

AGENT_SCENARIOS_PATH = "data/raw/agent_scenarios.csv"
OUTPUT_RUNS_PATH = "outputs/agent_runs.jsonl"
STATE_DIR = "state"


class SimulatedCrashError(Exception):
    """Raised to simulate an unexpected process crash during tool execution."""


def extract_refund_amount(status: str, final_answer: str, steps: List[Dict[str, Any]]) -> float:
    """Extracts or derives the refund amount (in INR) from tool executions or final answer."""
    # If the run ended in any fallback, no refund was authorized
    if status.startswith("FALLBACK") or status in ("DENIED", "NOT_FOUND"):
        return 0.0

    # 1. Check ledger-recording tools executed in steps (issue_refund / escalate_to_human)
    for s in reversed(steps):
        tool = s.get("tool") or s.get("tool_name")
        if tool in ("issue_refund", "escalate_to_human"):
            obs = s.get("observation", {})
            if isinstance(obs, dict) and "refund_inr" in obs:
                return round(float(obs["refund_inr"]), 2)
            args = s.get("arguments", {})
            if "refund_inr" in args:
                return round(float(args["refund_inr"]), 2)

    # 2. Extract from final answer text if present (e.g. "REFUND: INR 128.25 issued.")
    match = re.search(r"INR\s+([\d.]+)", final_answer)
    if match:
        return round(float(match.group(1)), 2)

    return 0.0


def save_checkpoint(
    run_id: str,
    scenario_id: str,
    messages: List[Dict[str, Any]],
    tokens_used: int,
    step_number: int,
    tools_executed: List[str],
    steps: List[Dict[str, Any]],
    llm_calls: int,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10.0,
):
    """Writes a persistent checkpoint file in state/<run_id>.json after tool observation."""
    os.makedirs(STATE_DIR, exist_ok=True)
    checkpoint_path = os.path.join(STATE_DIR, f"{run_id}.json")
    checkpoint_data = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "step_number": step_number,
        "tools_executed": tools_executed,
        "steps": steps,
        "llm_calls": llm_calls,
        "max_iterations": max_iterations,
        "token_limit": token_limit,
        "time_limit_seconds": time_limit_seconds,
    }
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)


def _execute_react_loop(
    run_id: str,
    scenario_id: str,
    messages: List[Dict[str, Any]],
    steps: List[Dict[str, Any]],
    tools_executed: List[str],
    llm_calls: int,
    tokens_used: int,
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10.0,
    crash_at_step: Optional[int] = None,
) -> Dict[str, Any]:
    """Shared ReAct loop engine for initial runs and resumed executions."""
    final_answer: Optional[str] = None
    status: Optional[str] = None
    start_time = time.monotonic()

    # Loop for remaining iterations
    while llm_calls < max_iterations:
        # Limit (c): At top of each iteration, check wall-clock timeout
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            final_answer = (
                f"FALLBACK_TIMEOUT: Wall-clock limit of {time_limit_seconds}s reached "
                f"(elapsed {elapsed:.4f}s); handing off to human support lead."
            )
            break

        resp = chat(messages, tools=TOOLS_SPEC)
        llm_calls += 1
        tokens_used += resp.usage.total_tokens

        # Limit (b): Cumulative token budget exceeded
        if tokens_used > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            final_answer = (
                f"FALLBACK_TOKEN_LIMIT: Cumulative token limit of {token_limit} exceeded "
                f"(accumulated {tokens_used}); handing off to human support lead."
            )
            break

        choice = resp.choices[0]
        msg = choice.message
        tool_calls = getattr(msg, "tool_calls", None)

        # If reply has no tool calls, its content is the final answer
        if not tool_calls:
            final_answer = msg.content or ""
            if ":" in final_answer:
                status = final_answer.split(":", 1)[0].strip()
            else:
                status = final_answer.strip()
            break

        # Otherwise append assistant message with tool calls
        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in tool_calls
            ],
        }
        messages.append(assistant_msg)

        # Execute each tool and append tool message
        for tc in tool_calls:
            if tc.name in TOOL_MAP:
                res = TOOL_MAP[tc.name](**tc.arguments)
            else:
                res = {"error": f"Tool '{tc.name}' not found."}

            tools_executed.append(tc.name)
            steps.append({
                "tool": tc.name,
                "tool_name": tc.name,
                "arguments": tc.arguments,
                "observation": res,
            })

            tool_msg: Dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tc.id,
                "name": tc.name,
                "content": json.dumps(res),
            }
            messages.append(tool_msg)

            # Checkpoint written after every tool observation is appended
            save_checkpoint(
                run_id=run_id,
                scenario_id=scenario_id,
                messages=messages,
                tokens_used=tokens_used,
                step_number=len(steps),
                tools_executed=tools_executed,
                steps=steps,
                llm_calls=llm_calls,
                max_iterations=max_iterations,
                token_limit=token_limit,
                time_limit_seconds=time_limit_seconds,
            )

            # Simulated crash trigger right after checkpoint is saved
            if crash_at_step is not None and len(steps) == crash_at_step:
                raise SimulatedCrashError(
                    f"Simulated process crash triggered right after step {crash_at_step} "
                    f"({tc.name}) checkpoint was saved."
                )

    # Limit (a): After max_iterations model calls without final answer
    if status is None and final_answer is None:
        status = "FALLBACK_MAX_ITERATIONS"
        final_answer = (
            f"FALLBACK_MAX_ITERATIONS: Maximum iterations ({max_iterations}) reached without final answer; "
            "handing off to human support lead."
        )

    refund_amount = extract_refund_amount(status, final_answer, steps)

    return {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "status": status,
        "final_answer": final_answer,
        "steps": steps,
        "tools_executed": tools_executed,
        "llm_calls": llm_calls,
        "tokens_used": tokens_used,
        "refund_amount": refund_amount,
    }


def run_agent(
    user_message: str,
    scenario_id: str = "",
    max_iterations: int = 6,
    token_limit: int = 6000,
    time_limit_seconds: float = 10.0,
    run_id: Optional[str] = None,
    crash_at_step: Optional[int] = None,
) -> Dict[str, Any]:
    """Executes the ReAct loop for a support ticket scenario with safety limits and checkpointing.

    Parameters:
        user_message: The customer request message.
        scenario_id: Identifier for the scenario (e.g. S1).
        max_iterations: Maximum allowed model interaction cycles (default 6).
        token_limit: Maximum cumulative tokens across model calls (default 6000).
        time_limit_seconds: Maximum wall-clock execution time (default 10s).
        run_id: Unique run identifier for checkpoint naming. Defaults to 'run_<scenario_id>'.
        crash_at_step: If specified, triggers a SimulatedCrashError immediately after this step.

    Returns:
        dict with run_id, scenario_id, status, final_answer, steps, tools_executed, llm_calls, tokens_used, refund_amount.
    """
    if run_id is None:
        run_id = f"run_{scenario_id}" if scenario_id else f"run_{int(time.time()*1000)}"

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": "You are a customer support AGENT helping users with refunds."},
        {"role": "user", "content": user_message},
    ]

    return _execute_react_loop(
        run_id=run_id,
        scenario_id=scenario_id,
        messages=messages,
        steps=[],
        tools_executed=[],
        llm_calls=0,
        tokens_used=0,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_at_step=crash_at_step,
    )


def resume_agent(run_id: str) -> Dict[str, Any]:
    """Loads a saved checkpoint from state/<run_id>.json and seamlessly continues the ReAct loop.

    Parameters:
        run_id: Unique run identifier whose checkpoint should be resumed.

    Returns:
        dict with completed run state and final outcomes.
    """
    checkpoint_path = os.path.join(STATE_DIR, f"{run_id}.json")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint for run_id '{run_id}' not found at {checkpoint_path}")

    with open(checkpoint_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenario_id = data.get("scenario_id", "")
    messages = data["messages"]
    tokens_used = data.get("tokens_used", 0)
    steps = data.get("steps", [])
    tools_executed = list(data.get("tools_executed", [s["tool"] for s in steps]))
    llm_calls = data.get("llm_calls", 0)
    max_iterations = data.get("max_iterations", 6)
    token_limit = data.get("token_limit", 6000)
    time_limit_seconds = data.get("time_limit_seconds", 10.0)

    return _execute_react_loop(
        run_id=run_id,
        scenario_id=scenario_id,
        messages=messages,
        steps=steps,
        tools_executed=tools_executed,
        llm_calls=llm_calls,
        tokens_used=tokens_used,
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        crash_at_step=None,
    )


def run_all_scenarios(
    scenarios_csv: str = AGENT_SCENARIOS_PATH,
    output_jsonl: str = OUTPUT_RUNS_PATH,
) -> List[Dict[str, Any]]:
    """Runs all scenarios from CSV, appends runs to JSONL, and returns results list."""
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)

    # Clear prior output if re-running
    if os.path.exists(output_jsonl):
        os.remove(output_jsonl)

    with open(scenarios_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        scenarios = list(reader)

    results = []
    print(f"Running {len(scenarios)} scenarios from {scenarios_csv}...\n")
    print(f"{'Scenario':<10} | {'Status':<25} | {'Refund (INR)':<14} | {'Expected INR':<14} | {'Calls':<6} | {'Tokens':<8}")
    print("-" * 88)

    for row in scenarios:
        sid = row["scenario_id"]
        msg = row["message"]
        expected_status = row["expected_outcome"]
        expected_refund = float(row["expected_refund_inr"])

        # Execute ReAct loop
        run_res = run_agent(user_message=msg, scenario_id=sid, run_id=f"run_{sid}")
        results.append(run_res)

        # Prepare JSON line with exact required fields
        line_data = {
            "scenario_id": sid,
            "status": run_res["status"],
            "final_answer": run_res["final_answer"],
            "llm_calls": run_res["llm_calls"],
            "tokens_used": run_res["tokens_used"],
            "steps": [
                {
                    "tool": s["tool"],
                    "tool_name": s["tool_name"],
                    "arguments": s["arguments"],
                    "observation": s["observation"],
                }
                for s in run_res["steps"]
            ],
            "refund_amount": run_res["refund_amount"],
        }

        # Append one JSON line per run
        with open(output_jsonl, mode="a", encoding="utf-8") as out_f:
            out_f.write(json.dumps(line_data) + "\n")

        print(
            f"{sid:<10} | {run_res['status']:<25} | {run_res['refund_amount']:<14.2f} | "
            f"{expected_refund:<14.2f} | {run_res['llm_calls']:<6} | {run_res['tokens_used']:<8}"
        )

        # Assertions against reference requirements
        if sid == "S7":
            assert run_res["status"] == "FALLBACK_MAX_ITERATIONS", (
                f"S7 status {run_res['status']} != FALLBACK_MAX_ITERATIONS"
            )
            assert run_res["llm_calls"] == 6, f"S7 model calls {run_res['llm_calls']} != 6"
            assert 2700 <= run_res["tokens_used"] <= 5500, (
                f"S7 tokens {run_res['tokens_used']} outside [2700, 5500]"
            )
        else:
            assert run_res["status"] == expected_status, (
                f"{sid}: status {run_res['status']} != expected {expected_status}"
            )
        assert abs(run_res["refund_amount"] - expected_refund) <= 0.01, (
            f"{sid}: refund {run_res['refund_amount']} != expected {expected_refund}"
        )
        if sid == "S1":
            assert run_res["llm_calls"] == 5, f"S1 model calls {run_res['llm_calls']} != 5"

    print("-" * 88)
    print(f"All 7 scenarios successfully verified and logged to {output_jsonl}!\n")
    return results


def run_safety_demos():
    """Runs and verifies the safety fallback demonstrations: token limit and timeout."""
    print("Running safety limit demonstrations...\n")
    s1_msg = "My order ORD-100048 reached me very late. I would like a refund."

    # Demo 1: S1 with token_limit=200
    res_token = run_agent(s1_msg, scenario_id="S1_TOKEN_LIMIT_DEMO", token_limit=200, run_id="demo_token_limit")
    print("--- Demo (b): Token Limit (token_limit=200) ---")
    print(f"Status:          {res_token['status']}")
    print(f"Model calls:     {res_token['llm_calls']}")
    print(f"Tokens used:     {res_token['tokens_used']}")
    print(f"Steps completed: {len(res_token['steps'])} ({[s['tool'] for s in res_token['steps']]})")
    print(f"Final answer:    {res_token['final_answer']}")
    assert res_token["status"] == "FALLBACK_TOKEN_LIMIT", f"Expected FALLBACK_TOKEN_LIMIT, got {res_token['status']}"
    assert res_token["llm_calls"] == 3, f"Expected 3 model calls, got {res_token['llm_calls']}"
    assert len(res_token["steps"]) == 2, f"Expected 2 completed steps, got {len(res_token['steps'])}"
    print("-> Verified: S1 stopped immediately without executing the 3rd requested tool.\n")

    # Demo 2: time_limit_seconds=0
    res_time = run_agent(s1_msg, scenario_id="TIMEOUT_DEMO", time_limit_seconds=0, run_id="demo_timeout")
    print("--- Demo (c): Timeout (time_limit_seconds=0) ---")
    print(f"Status:          {res_time['status']}")
    print(f"Model calls:     {res_time['llm_calls']}")
    print(f"Tokens used:     {res_time['tokens_used']}")
    print(f"Steps completed: {len(res_time['steps'])}")
    print(f"Final answer:    {res_time['final_answer']}")
    assert res_time["status"] == "FALLBACK_TIMEOUT", f"Expected FALLBACK_TIMEOUT, got {res_time['status']}"
    assert res_time["llm_calls"] == 0, f"Expected 0 model calls, got {res_time['llm_calls']}"
    assert len(res_time["steps"]) == 0, f"Expected 0 steps, got {len(res_time['steps'])}"
    print("-> Verified: Stopped before the first model call on wall-clock timeout.\n")

    return res_token, res_time


def run_resumable_demo() -> Dict[str, Any]:
    """Demonstrates crash recovery with S1: crashes after step 2, resumes, and finishes."""
    print("Running Resumable Agent Crash & Recovery Demonstration (Scenario S1)...\n")
    s1_msg = "My order ORD-100048 reached me very late. I would like a refund."
    run_id = "s1_crash_demo"

    # Part 1: Run with simulated crash right after 2nd tool execution
    pre_crash_tools: List[str] = []
    try:
        run_agent(s1_msg, scenario_id="S1", run_id=run_id, crash_at_step=2)
    except SimulatedCrashError as crash_err:
        print(f"[Simulated Crash Event]: {crash_err}")
        checkpoint_file = os.path.join(STATE_DIR, f"{run_id}.json")
        assert os.path.exists(checkpoint_file), f"Checkpoint file {checkpoint_file} must exist after crash"
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            ckpt_data = json.load(f)
        pre_crash_tools = ckpt_data["tools_executed"]
        print(f"  Checkpoint successfully saved at step {ckpt_data['step_number']}")
        print(f"  Tools executed pre-crash: {pre_crash_tools}\n")

    assert pre_crash_tools == ["get_order", "search_policy"], (
        f"Pre-crash tools {pre_crash_tools} != ['get_order', 'search_policy']"
    )

    # Part 2: Resume agent from checkpoint
    print(f"[Resume Event]: Calling resume_agent('{run_id}')...")
    resumed_res = resume_agent(run_id)

    print(f"  Resumed Status:       {resumed_res['status']}")
    print(f"  Resumed Final Answer: {resumed_res['final_answer']}")
    print(f"  Resumed Refund:       ₹{resumed_res['refund_amount']:.2f}")
    print(f"  Tools Executed Across Crash & Resume: {resumed_res['tools_executed']}")
    print(f"  Total LLM Calls:      {resumed_res['llm_calls']}\n")

    # Assertions
    expected_tools = ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed_res["tools_executed"] == expected_tools, (
        f"Executed tools {resumed_res['tools_executed']} != {expected_tools}"
    )
    assert resumed_res["status"] == "REFUND", f"Resumed status {resumed_res['status']} != REFUND"
    assert abs(resumed_res["refund_amount"] - 128.25) <= 0.01, (
        f"Resumed refund {resumed_res['refund_amount']} != 128.25"
    )
    print("-> Verified: Exactly 4 tool executions across crash and resume, none repeated, status REFUND (128.25).\n")

    return resumed_res


def print_markdown_table(
    results: List[Dict[str, Any]],
    demo_token: Dict[str, Any],
    demo_time: Dict[str, Any],
    resumed_demo: Dict[str, Any],
):
    """Prints markdown tables suitable for README.md Part D."""
    print("### ReAct Refund Agent Evaluation Table (All 7 Scenarios)\n")
    print("| Scenario ID | Status | Refund Amount (INR) | Model Calls | Tokens Used |")
    print("|---|---|---|---|---|")
    for r in results:
        print(
            f"| `{r['scenario_id']}` | **{r['status']}** | ₹{r['refund_amount']:.2f} | "
            f"{r['llm_calls']} | {r['tokens_used']} |"
        )
    print()

    print("### Safety Limit Fallback Demonstrations Table\n")
    print("| Demo Run | Scenario | Condition / Limit | Resulting Status | Model Calls | Steps Completed | Tokens Used | Final Answer / Human Handoff |")
    print("|---|---|---|---|---|---|---|---|")
    print(
        f"| Max Iterations | `S7` | `max_iterations=6` | **{results[-1]['status']}** | "
        f"{results[-1]['llm_calls']} | {len(results[-1]['steps'])} | {results[-1]['tokens_used']} | {results[-1]['final_answer']} |"
    )
    print(
        f"| Token Limit | `S1` | `token_limit=200` | **{demo_token['status']}** | "
        f"{demo_token['llm_calls']} | {len(demo_token['steps'])} | {demo_token['tokens_used']} | {demo_token['final_answer']} |"
    )
    print(
        f"| Timeout Limit | `S1` | `time_limit_seconds=0` | **{demo_time['status']}** | "
        f"{demo_time['llm_calls']} | {len(demo_time['steps'])} | {demo_time['tokens_used']} | {demo_time['final_answer']} |"
    )
    print()

    print("### Resumable Crash & Recovery Demonstration (Scenario S1)\n")
    print("| Phase | Step | Tool Executed | Status / Event |")
    print("|---|---|---|---|")
    print("| Pre-Crash | 1 | `get_order` | Checkpoint written to `state/s1_crash_demo.json` |")
    print("| Pre-Crash | 2 | `search_policy` | Checkpoint written to `state/s1_crash_demo.json` |")
    print("| Crash Trigger | - | - | Simulated crash raised immediately after step 2 checkpoint |")
    print("| Resume Call | - | - | `resume_agent('s1_crash_demo')` called from persisted state |")
    print("| Post-Resume | 3 | `calculate_refund` | Checkpoint updated in `state/s1_crash_demo.json` |")
    print("| Post-Resume | 4 | `issue_refund` | Checkpoint updated in `state/s1_crash_demo.json` |")
    print(f"| Final State | - | - | Status **{resumed_demo['status']}** (₹{resumed_demo['refund_amount']:.2f}) |")
    print()


if __name__ == "__main__":
    reset()
    all_res = run_all_scenarios()
    demo_tok, demo_tim = run_safety_demos()
    resumed_res = run_resumable_demo()
    print_markdown_table(all_res, demo_tok, demo_tim, resumed_res)
