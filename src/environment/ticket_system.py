"""Mock IT ticketing system. Deterministic in-memory store, reset per scenario."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class Priority(Enum):
    P1_CRITICAL = "P1"  # 1 hour SLA
    P2_HIGH = "P2"  # 4 hour SLA
    P3_NORMAL = "P3"  # 24 hour SLA


class Status(Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    ESCALATED = "escalated"
    RESOLVED = "resolved"


@dataclass
class Ticket:
    ticket_id: str
    employee_id: str
    subject: str
    priority: Priority
    status: Status = Status.OPEN
    notes: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class TicketSystem:
    def __init__(self):
        self._tickets: dict[str, Ticket] = {}
        self._next_id = 1

    def create_ticket(self, employee_id: str, subject: str, priority: str) -> dict:
        ticket_id = f"TKT-{self._next_id:04d}"
        self._next_id += 1
        self._tickets[ticket_id] = Ticket(
            ticket_id=ticket_id,
            employee_id=employee_id,
            subject=subject,
            priority=Priority(priority),
        )
        return self._tickets[ticket_id].__dict__

    def update_status(self, ticket_id: str, status: str, note: str | None = None, priority: str | None = None) -> dict:
        ticket = self._require(ticket_id)
        ticket.status = Status(status)
        if priority:
            ticket.priority = Priority(priority)
        if note:
            ticket.notes.append(note)
        return ticket.__dict__

    def escalate(self, ticket_id: str, reason: str) -> dict:
        ticket = self._require(ticket_id)
        ticket.status = Status.ESCALATED
        ticket.notes.append(f"ESCALATED: {reason}")
        return ticket.__dict__

    def get(self, ticket_id: str) -> dict:
        return self._require(ticket_id).__dict__

    def list_by_employee(self, employee_id: str) -> list[dict]:
        return [t.__dict__ for t in self._tickets.values() if t.employee_id == employee_id]

    def _require(self, ticket_id: str) -> Ticket:
        if ticket_id not in self._tickets:
            raise KeyError(f"No such ticket: {ticket_id}")
        return self._tickets[ticket_id]
