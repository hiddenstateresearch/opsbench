"""LLM-based helpdesk agent under test. Uses a local Ollama model's tool-calling
loop against the mock Environment — no paid API, runs entirely on local hardware.
This is the system being benchmarked (Experiment: policy adherence + task success
under a new internal-enterprise-ops vertical).
"""
import json
import re

import ollama

from environment import Environment, to_ollama_tools
from policy import POLICY_TEXT

MAX_TOOL_ITERS = 8
OLLAMA_TOOLS = to_ollama_tools()

# Granite 3.1 Dense writes <tool_call>[...] as plain text, which its Ollama template
# (expecting <|tool_call|>) never parses; no other evaluated model emits this format.
_PROSE_TOOL_CALL_RE = re.compile(r"<tool_call>(\[.*?\])", re.DOTALL)


def _parse_prose_tool_calls(content: str) -> list[dict]:
    calls = []
    for match in _PROSE_TOOL_CALL_RE.finditer(content or ""):
        try:
            for item in json.loads(match.group(1)):
                calls.append({"function": {"name": item["name"], "arguments": item["arguments"]}})
        except (json.JSONDecodeError, KeyError, TypeError):
            continue  # malformed call; leave it unparsed rather than crash the run
    return calls


class HelpdeskAgent:
    def __init__(self, env: Environment, model: str = "llama3.1:8b"):
        self.env = env
        self.model = model
        self.messages: list[dict] = [{"role": "system", "content": POLICY_TEXT}]
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def respond_to_user(self, user_message: str) -> str:
        """Feed a user turn in, run the tool-calling loop, return the agent's reply text."""
        self.messages.append({"role": "user", "content": user_message})

        for _ in range(MAX_TOOL_ITERS):
            response = ollama.chat(
                model=self.model,
                messages=self.messages,
                tools=OLLAMA_TOOLS,
                # Caps a single generation call so a model that falls into a
                # repetition loop without emitting a stop token can't stall a run
                # for tens of minutes (and can't silently inflate wall_clock_s).
                # 4096, not 1024: reasoning/thinking models (e.g. qwen3.6) can spend
                # 1000+ tokens on an internal thinking trace before any visible
                # content or tool call -- 1024 truncated those to empty replies.
                options={"num_predict": 4096},
            )
            msg = response["message"]
            # Ollama can return these keys present but set to None (not just absent),
            # which `.get(key, 0)` does not guard against.
            self.usage["input_tokens"] += response.get("prompt_eval_count") or 0
            self.usage["output_tokens"] += response.get("eval_count") or 0

            self.messages.append(msg)

            tool_calls = msg.get("tool_calls") or _parse_prose_tool_calls(msg.get("content"))
            if not tool_calls:
                return msg.get("content", "")

            for call in tool_calls:
                result = self.env.dispatch(call["function"]["name"], call["function"]["arguments"])
                self.messages.append({"role": "tool", "content": str(result)})

        return "[agent exceeded tool-call budget without a final reply]"
