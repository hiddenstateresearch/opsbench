"""Mock employee directory + identity verification. Deterministic, seeded per scenario."""
from dataclasses import dataclass


@dataclass
class Employee:
    employee_id: str
    name: str
    role: str
    manager_id: str | None
    verification_code: str  # simulates e.g. a code sent to a registered device


class UserDirectory:
    def __init__(self, employees: dict[str, Employee]):
        self._employees = employees
        self._verified_this_session: set[str] = set()
        self._password_reset_this_session: set[str] = set()

    def lookup(self, employee_id: str) -> dict:
        emp = self._require(employee_id)
        return {"employee_id": emp.employee_id, "name": emp.name, "role": emp.role}

    def verify_identity(self, employee_id: str, verification_code: str) -> dict:
        emp = self._require(employee_id)
        ok = verification_code.strip() == emp.verification_code
        if ok:
            self._verified_this_session.add(employee_id)
        return {"verified": ok}

    def is_verified(self, employee_id: str) -> bool:
        return employee_id in self._verified_this_session

    def reset_password(self, employee_id: str) -> dict:
        """Actually execute a password reset. Enforces the verification policy at the
        tool level too (defense in depth) rather than relying solely on the agent's
        own judgment -- refuses if the employee hasn't passed verify_identity first.
        """
        self._require(employee_id)
        if not self.is_verified(employee_id):
            return {"reset": False, "error": "identity not verified in this session"}
        self._password_reset_this_session.add(employee_id)
        return {"reset": True, "employee_id": employee_id}

    def was_password_reset(self, employee_id: str) -> bool:
        return employee_id in self._password_reset_this_session

    def requires_manager_approval(self, employee_id: str, cost: float, threshold: float = 500.0) -> bool:
        return cost > threshold

    def _require(self, employee_id: str) -> Employee:
        if employee_id not in self._employees:
            raise KeyError(f"No such employee: {employee_id}")
        return self._employees[employee_id]
