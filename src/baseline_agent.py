"""Trivial, non-LLM scripted baseline: always files a ticket, never resets a
password, and escalates anything outside a tiny hardcoded set of KB-resolvable
keywords. No model, no tool-calling loop -- deterministic rules only. Exists to
give pass rate / pass^k an interpretable floor: how much of each LLM agent's
score a fixed, capability-free policy would already get for free.
"""
import re

KB_KEYWORDS = ("printer", "wifi", "wi-fi", "network")


class ScriptedBaselineAgent:
    def __init__(self, env, model: str | None = None):
        self.env = env
        self.employee_id: str | None = None
        self.ticket_id: str | None = None
        self.is_kb_issue: bool | None = None  # decided once, from the first message seen
        self.kb_suggested = False
        self.escalated = False
        self.usage = {"input_tokens": 0, "output_tokens": 0}  # zero-cost by construction

    def respond_to_user(self, user_message: str) -> str:
        if self.is_kb_issue is None:
            self.is_kb_issue = any(k in user_message.lower() for k in KB_KEYWORDS)

        if self.ticket_id is None:
            # Keep retrying extraction every turn (not just the first) -- personas
            # sometimes state the employee ID a turn or two after the opening message.
            match = re.search(r"\bE\d+\b", user_message)
            if not match:
                return "Could you confirm your employee ID so I can open a ticket?"
            self.employee_id = match.group(0)
            lookup = self.env.dispatch("lookup_employee", {"employee_id": self.employee_id})
            if "error" in lookup:
                return "I can't find that employee record under that ID -- could you double check it?"
            ticket = self.env.dispatch("create_ticket", {
                "employee_id": self.employee_id,
                "subject": "Helpdesk request",
                "priority": "P2",
            })
            self.ticket_id = ticket.get("ticket_id")

        if self.escalated:
            return "This has already been escalated to a human technician."

        if self.is_kb_issue and not self.kb_suggested:
            self.kb_suggested = True
            return ("Please try turning the device off and on again, or unplugging "
                    "and replugging it. Let me know if that resolves it.")
        if self.is_kb_issue and self.kb_suggested:
            self.env.dispatch("update_status", {"ticket_id": self.ticket_id, "status": "resolved"})
            return "Glad that worked -- I've marked your ticket resolved."

        self.escalated = True
        self.env.dispatch("escalate_ticket", {
            "ticket_id": self.ticket_id, "reason": "outside trivial baseline scope",
        })
        return "I'm escalating this to a human technician."
