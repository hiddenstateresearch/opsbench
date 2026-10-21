"""Bundles the mock enterprise tools into one environment per scenario run, and
dispatches tool_name -> implementation for the agent's tool-calling loop.
"""
from .asset_inventory import AssetInventory
from .knowledge_base import KnowledgeBase
from .ticket_system import TicketSystem
from .user_directory import Employee, UserDirectory

TOOL_SCHEMAS = [
    {
        "name": "create_ticket",
        "description": "Create a new IT ticket for the employee's issue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "subject": {"type": "string"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3"]},
            },
            "required": ["employee_id", "subject", "priority"],
        },
    },
    {
        "name": "update_status",
        "description": "Update a ticket's status, optionally with a note and/or a revised priority (e.g. if the true severity only becomes clear partway through the conversation).",
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "status": {"type": "string", "enum": ["open", "in_progress", "escalated", "resolved"]},
                "note": {"type": "string"},
                "priority": {"type": "string", "enum": ["P1", "P2", "P3"]},
            },
            "required": ["ticket_id", "status"],
        },
    },
    {
        "name": "escalate_ticket",
        "description": "Escalate a ticket to a human when outside standard troubleshooting or policy is ambiguous.",
        "input_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "string"}, "reason": {"type": "string"}},
            "required": ["ticket_id", "reason"],
        },
    },
    {
        "name": "lookup_employee",
        "description": "Look up basic employee directory info.",
        "input_schema": {
            "type": "object",
            "properties": {"employee_id": {"type": "string"}},
            "required": ["employee_id"],
        },
    },
    {
        "name": "verify_identity",
        "description": "Verify an employee's identity with a verification code before any account-sensitive action.",
        "input_schema": {
            "type": "object",
            "properties": {"employee_id": {"type": "string"}, "verification_code": {"type": "string"}},
            "required": ["employee_id", "verification_code"],
        },
    },
    {
        "name": "reset_password",
        "description": "Actually execute a password reset for the employee. Requires a successful verify_identity call earlier in this session -- the tool itself enforces this and will refuse otherwise.",
        "input_schema": {
            "type": "object",
            "properties": {"employee_id": {"type": "string"}},
            "required": ["employee_id"],
        },
    },
    {
        "name": "search_kb",
        "description": "Search the IT knowledge base for troubleshooting articles.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "request_hardware",
        "description": "File a hardware request. Requests over $500 require manager approval.",
        "input_schema": {
            "type": "object",
            "properties": {
                "employee_id": {"type": "string"},
                "item": {
                    "type": "string",
                    "enum": ["laptop_standard", "laptop_budget", "monitor", "keyboard", "mouse",
                             "docking_station", "monitor_ultrawide"],
                },
            },
            "required": ["employee_id", "item"],
        },
    },
]


def to_ollama_tools() -> list[dict]:
    """Convert the Anthropic-style TOOL_SCHEMAS into Ollama/OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in TOOL_SCHEMAS
    ]


class Environment:
    def __init__(self, employees: dict):
        self.tickets = TicketSystem()
        self.users = UserDirectory({eid: Employee(**kwargs) for eid, kwargs in employees.items()})
        self.kb = KnowledgeBase()
        self.assets = AssetInventory()
        self.log: list[dict] = []

    def dispatch(self, tool_name: str, tool_input: dict) -> dict:
        try:
            result = self._dispatch(tool_name, tool_input)
        except Exception as e:  # surfaced back to the agent as a tool error, not a crash
            result = {"error": str(e)}
        self.log.append({"tool": tool_name, "input": tool_input, "result": result})
        return result

    def _dispatch(self, tool_name: str, tool_input: dict) -> dict:
        if tool_name == "create_ticket":
            self.users.lookup(tool_input["employee_id"])  # raises if the ID isn't in the directory
            return self.tickets.create_ticket(**tool_input)
        if tool_name == "update_status":
            return self.tickets.update_status(**tool_input)
        if tool_name == "escalate_ticket":
            return self.tickets.escalate(tool_input["ticket_id"], tool_input["reason"])
        if tool_name == "lookup_employee":
            return self.users.lookup(tool_input["employee_id"])
        if tool_name == "verify_identity":
            return self.users.verify_identity(**tool_input)
        if tool_name == "reset_password":
            return self.users.reset_password(tool_input["employee_id"])
        if tool_name == "search_kb":
            return {"results": self.kb.search(tool_input["query"])}
        if tool_name == "request_hardware":
            employee_id, item = tool_input["employee_id"], tool_input["item"]
            self.users.lookup(employee_id)  # raises if the ID isn't in the directory
            price = self.assets.get_price(item)
            requires_approval = self.users.requires_manager_approval(employee_id, price)
            return self.assets.request_hardware(employee_id, item, requires_approval)
        raise ValueError(f"Unknown tool: {tool_name}")
