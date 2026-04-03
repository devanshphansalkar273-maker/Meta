from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field

class Decision(str, Enum):
    ALLOW = "ALLOW"
    FLAG = "FLAG"
    REMOVE = "REMOVE"
    ESCALATE = "ESCALATE"

class ContentCategory(str, Enum):
    SPAM = "SPAM"
    HATE_SPEECH = "HATE_SPEECH"
    MISINFORMATION = "MISINFORMATION"
    HARASSMENT = "HARASSMENT"
    SAFE = "SAFE"

class UserMetadata(BaseModel):
    user_id: str = Field(..., description="Stable user tracking identity.")
    timestamp: str = Field(..., description="When the post was made (ISO 8601).")
    reports_count: int = Field(default=0, description="Number of independent users who reported the post.")
    author_trust_score: float = Field(..., ge=0.0, le=1.0, description="Dynamically updated trust score.")
    account_age_days: int = Field(..., ge=0, description="How long the author's account has existed.")
    virality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Momentum metrics of recent engagements.")

class ModerationObservation(BaseModel):
    post_id: str = Field(..., description="Unique ID for the content piece.")
    post_body: str = Field(..., description="The actual text of the post.")
    metadata: UserMetadata = Field(..., description="Information regarding the poster and engagement.")
    context: List[str] = Field(default_factory=list, description="Historical actions or flags associated with the author.")

class ModerationAction(BaseModel):
    decision: Decision = Field(..., description="The definitive action to take on the post.")
    content_category: ContentCategory = Field(..., description="The AI's prediction of the content subset classification.")
    reasoning: str = Field(..., description="Brief reasoning of why this decision was made.")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="The agent's confidence in its decision.")

class ModerationReward(BaseModel):
    score: float = Field(..., description="Reward score based on cost-sensitive and reputation matrices.")
    reason: str = Field(..., description="Explanation spanning partial confidence and categorical matching.")
