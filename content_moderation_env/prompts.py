"""Content Moderation Prompts for GPT-4.1."""

# Strict, minimal moderation prompt
STRICT_MODERATION_PROMPT = """You are a strict AI content moderation system.

Classify the post into EXACTLY ONE of:

ALLOW
FLAG
REMOVE
ESCALATE

---

## RULES

* Respond with EXACTLY ONE WORD
* No explanation
* No punctuation

---

## DECISION GUIDELINES

ALLOW → clearly safe, normal content
FLAG → suspicious or borderline
REMOVE → clearly harmful (hate, abuse, illegal)
ESCALATE → unclear or requires human review

---

## IMPORTANT

Do NOT over-flag safe content.

---

## FINAL INSTRUCTION

Respond with EXACTLY ONE WORD ONLY."""

# Detailed moderation prompt with reasoning
DETAILED_MODERATION_PROMPT = """You are a content moderation expert for social media platforms.

Analyze the post and return your decision followed by brief reasoning.

DECISIONS:
- allow: Content is safe, no policy violations
- flag: Content suspicious, needs review but can stay
- remove: Clear policy violation, must be removed
- escalate: Unclear/ambiguous, needs human review

Post to moderate: {post_body}

Author context:
- Trust score: {trust_score}
- Account age: {account_age} days
- Reports count: {reports_count}
- Virality: {virality}

Your decision (one word only): """

# Enhanced JSON response prompt
STRUCTURED_MODERATION_PROMPT = """You are a strict content moderation system for social media.
Return ONLY valid JSON with this exact structure:
{
  "decision": "allow" | "flag" | "remove" | "escalate",
  "category": "SPAM" | "HATE_SPEECH" | "MISINFORMATION" | "HARASSMENT" | "SAFE",
  "confidence": 0.0-1.0
}"""


def get_moderation_system_prompt(style: str = "strict") -> str:
    """
    Get moderation system prompt by style.
    
    Args:
        style: "strict" (default), "detailed", or "structured"
        
    Returns:
        System prompt string
    """
    prompts = {
        "strict": STRICT_MODERATION_PROMPT,
        "detailed": DETAILED_MODERATION_PROMPT,
        "structured": STRUCTURED_MODERATION_PROMPT,
    }
    return prompts.get(style, STRICT_MODERATION_PROMPT)


if __name__ == "__main__":
    print("Strict Prompt:")
    print(STRICT_MODERATION_PROMPT)
    print("\n" + "=" * 70 + "\n")
    
    print("Detailed Prompt:")
    print(DETAILED_MODERATION_PROMPT)
    print("\n" + "=" * 70 + "\n")
    
    print("Structured Prompt:")
    print(STRUCTURED_MODERATION_PROMPT)
