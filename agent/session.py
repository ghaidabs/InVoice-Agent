from dataclasses import dataclass, field
from typing import Optional

@dataclass
class UserContext:
    user_id: int
    company_id: int
    role: str
    company_slug: str  
    
@dataclass
class SessionState:
    user: UserContext
    pending_draft: Optional[dict] = None
    last_intent: Optional[str] = None
    confirmation_pending: bool = False
    awaiting_field: Optional[str] = None       
    pending_transcript: Optional[str] = None  