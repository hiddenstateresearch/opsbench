"""Mock hardware/asset request system."""
from dataclasses import dataclass, field


@dataclass
class HardwareRequest:
    request_id: str
    employee_id: str
    item: str
    cost: float
    approved: bool = False
    manager_approval_pending: bool = False


class AssetInventory:
    CATALOG = {
        "laptop_standard": 800.0,
        "laptop_budget": 450.0,
        "monitor": 250.0,
        "keyboard": 60.0,
        "mouse": 30.0,
        "docking_station": 180.0,
        "monitor_ultrawide": 500.0,
    }

    def __init__(self):
        self._requests: dict[str, HardwareRequest] = {}
        self._next_id = 1

    def get_price(self, item: str) -> float:
        if item not in self.CATALOG:
            raise KeyError(f"Unknown catalog item: {item}. Available: {list(self.CATALOG)}")
        return self.CATALOG[item]

    def request_hardware(self, employee_id: str, item: str, requires_manager_approval: bool) -> dict:
        request_id = f"REQ-{self._next_id:04d}"
        self._next_id += 1
        req = HardwareRequest(
            request_id=request_id,
            employee_id=employee_id,
            item=item,
            cost=self.get_price(item),
            approved=not requires_manager_approval,
            manager_approval_pending=requires_manager_approval,
        )
        self._requests[request_id] = req
        return req.__dict__

    def list_by_employee(self, employee_id: str) -> list[dict]:
        return [r.__dict__ for r in self._requests.values() if r.employee_id == employee_id]
