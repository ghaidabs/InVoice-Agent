from dataclasses import dataclass, field
from typing import Optional

@dataclass
class UserContext:
    user_id: int
    company_id: int
    role: str
    company_slug: str  # needed for invoice number generation

@dataclass
class SessionState:
    user: UserContext
    pending_draft: Optional[dict] = None
    last_intent: Optional[str] = None
    confirmation_pending: bool = False