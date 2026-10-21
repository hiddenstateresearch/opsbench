"""Local-Ollama-driven simulated employee, following a scenario persona. Mirrors
tau-bench's user-simulator pattern: a separate LLM plays the human side of the
conversation so the benchmark is dialog-driven rather than single-shot.
"""
import ollama

SYSTEM_TEMPLATE = """You are role-playing as an employee contacting IT helpdesk. Stay
in character and follow this persona exactly:

{persona}

Keep replies short and natural, like a real chat message. When your issue has been
resolved (or you've been clearly told it's escalated/pending and you acknowledge
that), end your message with the exact token [END_CHAT]. Do not mention that you are
an AI or that this is a simulation.
"""


class UserSimulator:
    def __init__(self, persona: str, model: str = "llama3.1:8b"):
        self.model = model
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_TEMPLATE.format(persona=persona)}]

    def opening_message(self) -> str:
        return self.reply_to(None)

    def reply_to(self, agent_message: str | None) -> str:
        if agent_message is not None:
            self.messages.append({"role": "user", "content": agent_message})
        else:
            self.messages.append(
                {"role": "user", "content": "(The chat has just started. Send your opening message.)"}
            )
        response = ollama.chat(
            model=self.model,
            messages=self.messages,
            # Same runaway-generation guard as agent.py; user turns are supposed
            # to be short chat replies anyway.
            options={"num_predict": 512},
        )
        text = response["message"]["content"]
        self.messages.append({"role": "assistant", "content": text})
        return text

    @staticmethod
    def is_done(message: str) -> bool:
        return "[END_CHAT]" in message
