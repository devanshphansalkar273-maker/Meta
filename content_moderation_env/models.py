from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class Decision(str, Enum):
    ALLOW = "ALLOW"
    FLAG = "FLAG"
    REMOVE = "REMOVE"
    ESCALATE = "ESCALATE"


class ContentCategory(str, Enum):
    SAFE = "SAFE"
    SPAM = "SPAM"
    TOXIC = "TOXIC"
    HARASSMENT = "HARASSMENT"
    HATE_SPEECH = "HATE_SPEECH"
    VIOLENCE = "VIOLENCE"
    SEXUAL = "SEXUAL"
    SELF_HARM = "SELF_HARM"
    MISINFORMATION = "MISINFORMATION"


class UserMetadata(BaseModel):
    user_id: str
    timestamp: Optional[str] = None
    reports_count: int = 0
    author_trust_score: float = 0.5
    account_age_days: int = 0
    virality_score: float = 0.0
    active_global_event: str = "None"
    temporary_rule: str = "Standard policy applies"
    user_appeal_statement: str = "I did nothing wrong!"
    media_vision_tags: List[str] = Field(default_factory=list)
    visual_text_mismatch_flag: bool = False


class ModerationObservation(BaseModel):
    post_id: str
    post_body: str
    metadata: UserMetadata
    context: List[str] = Field(default_factory=list)


class ModerationAction(BaseModel):
    decision: Decision
    content_category: ContentCategory
    reasoning: str = ""
    confidence_score: float


class ModerationReward(BaseModel):
    score: float
    reason: str


class ModerationState(BaseModel):
    step_count: int
    current_index: int
    task: str
    done: bool
