"""Mock IT knowledge base. Simple keyword search over a fixed article set."""

ARTICLES = [
    {
        "id": "KB-001",
        "title": "How to reset your password",
        "keywords": ["password", "password reset", "forgot", "login", "log in",
                     "locked", "lockout", "unlock", "account locked"],
        "body": (
            "Password resets require identity verification via verify_identity before "
            "any reset can be issued. Never reset a password without a successful "
            "verification in this session. An account locked out after too many failed "
            "attempts is cleared the same way: verify identity, then reset the password."
        ),
    },
    {
        "id": "KB-002",
        "title": "VPN connection troubleshooting",
        "keywords": ["vpn", "connect", "network", "remote"],
        "body": (
            "1) Confirm the client is updated to the latest version. 2) Confirm the "
            "employee's account is active. 3) If issue persists after basic steps, "
            "escalate to network engineering."
        ),
    },
    {
        "id": "KB-003",
        "title": "Requesting new hardware",
        "keywords": ["laptop", "hardware", "device", "monitor", "replacement",
                     "mouse", "keyboard", "peripheral", "docking", "docking station"],
        "body": (
            "Hardware requests under $500 may be approved directly. Requests over $500 "
            "require manager approval before the order can be placed."
        ),
    },
    {
        "id": "KB-004",
        "title": "Printer not working",
        "keywords": ["printer", "print", "printing"],
        "body": "1) Check printer is powered and connected. 2) Restart print spooler. 3) Reinstall printer driver if issue persists.",
    },
    {
        "id": "KB-005",
        "title": "Wifi connectivity troubleshooting",
        "keywords": ["wifi", "wi-fi", "wireless"],
        "body": (
            "1) Confirm the device is connecting to the corporate network, not a guest "
            "network. 2) Forget and re-add the network. 3) Restart the device's wifi "
            "adapter. If issue persists after basic steps, escalate to network engineering."
        ),
    },
    {
        "id": "KB-006",
        "title": "Reporting suspicious activity or phishing",
        "keywords": ["phishing", "suspicious", "hacked", "breach", "malware", "security incident"],
        "body": (
            "Do not attempt to remediate a suspected security incident yourself. "
            "Escalate immediately to the security team via escalate_ticket with priority P1."
        ),
    },
    {
        "id": "KB-007",
        "title": "Desktop wallpaper or personalization settings not saving",
        "keywords": ["wallpaper", "desktop background", "background image", "personalization", "theme"],
        "body": (
            "A custom wallpaper or theme reverting after every restart is caused by the "
            "managed-desktop policy re-applying the default image on login. This is "
            "expected behavior, not a fault -- explain this to the employee, log the "
            "ticket for the desktop team's awareness, and resolve it; no escalation needed."
        ),
    },
]


class KnowledgeBase:
    def search(self, query: str) -> list[dict]:
        q = query.lower()
        return [a for a in ARTICLES if any(kw in q for kw in a["keywords"])]
