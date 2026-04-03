# ContentGuard: RL Environment for AI Content Moderation

## 1. Overview

**Environment Name**: `ContentGuard-v1`
**Environment Type**: Partially Observable Markov Decision Process (POMDP) with stochastic transitions
**Framework**: OpenAI Gymnasium-compatible
**Target Users**: AI safety researchers, moderation system developers, RL agent evaluators

### Core Problem
Real-time moderation of social media posts where:
- Ground truth labels are not immediately available (delayed feedback)
- Report streams arrive asynchronously over time
- Actions have cascading effects on user behavior
- Trade-offs exist between precision, recall, and response latency

---

## 2. Environment Dynamics

### 2.1 World State

```python
@dataclass
class WorldState:
    # Queue of pending posts (FIFO, max 50)
    pending_posts: deque[Post]

    # Currently processing post
    current_post: Post | None

    # Post under active review (may differ from current_post due to async reports)
    active_review_post: Post | None

    # Accumulated reports for active_review_post
    accumulated_reports: list[Report]

    # Time since active_review_post arrived (steps)
    review_duration: int

    # System-level metrics
    queue_depth: int                    # Posts waiting in queue
    avg_processing_time: float          # Rolling average
    false_positive_rate: float          # Historical FP rate
    false_negative_rate: float          # Historical FN rate
    budget_remaining: float            # Computational/budget for escalations

    # External factors
    time_of_day: int                    # 0-23 hours
    day_of_week: int                    # 0-6
    trending_topic_factor: float       # 0.5-2.0 multiplier on report rate
```

### 2.2 Post Entity

```python
@dataclass
class Post:
    post_id: str
    author_id: str
    text: str
    media_type: str                     # "text" | "image" | "video" | "link"
    created_at: float                   # Unix timestamp
    initial_report_count: int          # Reports available at submission time
    final_report_count: int             # Ground truth total reports (revealed later)
    ground_truth_label: Label          # "toxic" | "spam" | "safe"
    metadata: PostMetadata

@dataclass
class PostMetadata:
    author_account_age_days: int
    author_post_count: int
    author_ban_history: int
    author_verified: bool
    text_language: str
    text_has_emoji: bool
    text_has_url: bool
    post_engagement_rate: float         # Mock engagement metric
    is_edit: bool
    is_reply: bool
    reply_to_author_follower_count: int
```

### 2.3 Report Entity

```python
@dataclass
class Report:
    report_id: str
    post_id: str
    reporter_id: str
    report_type: str                    # "hate_speech" | "spam" | "harassment" | "misinformation" | "other"
    reporter_trust_score: float         # 0.0-1.0
    created_at: float
    is_auto_report: bool                # Bot detection flag
```

---

## 3. Observation Space

### 3.1 Full Observation (Research Mode)

```python
class ObservationSpace:
    """
    Box space with shape (78,) for full observability research.
    All values normalized to [0, 1] or [-1, 1].
    """
    def __init__(self):
        self.shape = (78,)
        self.dtype = np.float32

    # Text Features (32 dims via TF-IDF or embedding projection)
    self.text_embedding: np.array(32)      # Pre-computed text embedding

    # Immediate Report Features (10 dims)
    self.report_count: float                # 0-50 normalized
    self.avg_reporter_trust: float          # 0-1
    self.auto_report_ratio: float           # 0-1
    self.report_type_vector: np.array(5)     # one-hot-ish: hate, spam, harass, misinfo, other
    self.report_velocity: float             # Reports per minute (clipped to 10)

    # Author Features (10 dims)
    self.author_account_age: float           # Log-scaled days
    self.author_post_count: float            # Log-scaled
    self.author_ban_history: float           # 0-5 normalized
    self.author_verified: float              # 0 or 1
    self.author_trust_score: float          # Computed rolling score

    # Post Features (8 dims)
    self.media_type: np.array(4)            # one-hot: text, image, video, link
    self.text_language_known: float         # 0 or 1
    self.has_url: float                      # 0 or 1
    self.is_reply: float                     # 0 or 1
    self.is_edit: float                      # 0 or 1
    self.engagement_rate: float             # 0-1 normalized

    # Queue/System Features (6 dims)
    self.queue_depth: float                 # 0-50 normalized
    self.review_duration: float             # Steps since assigned (clipped 0-20)
    self.time_of_day: float                 # 0-23 normalized
    self.day_of_week: float                  # 0-6 normalized
    self.trending_factor: float              # 0.5-2.0 normalized to 0-1
    self.budget_remaining: float             # 0-1

    # Historical Performance Features (4 dims)
    self.fp_rate_rolling: float             # Last 100 decisions
    self.fn_rate_rolling: float             # Last 100 decisions
    self.avg_precision_rolling: float      # Last 100 decisions
    self.decision_confidence: float         # Agent's own confidence (if applicable)
```

### 3.2 Partial Observation (Production Mode)

```python
class PartialObservationSpace:
    """
    Realistic production observation - mimics actual API.
    Post metadata arrives first, reports stream in over time.
    """
    def __init__(self):
        self.shape = (52,)  # Fewer features, more realistic

    # Initial post data (available at submission)
    self.post_embedding: np.array(32)       # Same text embedding
    self.media_type: np.array(4)
    self.has_url: float
    self.is_reply: float
    self.author_trust_score: float

    # Reports available "now" (delayed from submission)
    self.report_count: float
    self.avg_trust: float
    self.report_vector: np.array(5)
    self.report_velocity: float

    # System state
    self.queue_depth: float
    self.review_duration: float

    # Delayed features (NOT available until later)
    # - Author ban history (revealed after 24h delay in real system)
    # - Full author history (revealed on first author action)
    # - Final report count (revealed after 7 days in real system)
```

### 3.3 Observation Delay Simulation

```python
"""
In production, reports arrive with realistic delays:
- Immediate (0-30s): Auto-reports from bot detection
- Short (30s-5min): Engaged users flagging
- Medium (5min-1hr): Moderation volunteers
- Long (1-24hr): User reports, escalations
- Very Long (24h-7d): Deep reviews, appeals

This creates a POMDP where optimal action depends on:
1. Whether to wait for more reports
2. Whether to escalate (delay but more info)
3. Immediate action vs batch processing
"""
```

---

## 4. Action Space

### 4.1 Discrete Action Space

```python
class ActionSpace(gym.spaces.Discrete):
    """
    8 discrete actions representing moderation decisions.
    """
    ACTIONS = {
        0: "ALLOW",              # Approve immediately, no flag
        1: "FLAG_TICKET",        # Add to review queue (human)
        2: "FLAG_AUTO",          # Flag for automated secondary check
        3: "REMOVE_IMMEDIATE",   # Remove without appeal
        4: "REMOVE_WARN",        # Remove with warning to author
        5: "ESCALATE_HIGH",      # Urgent escalation to senior reviewer
        6: "ESCALATE_MISINFO",   # Specialized misinformation team
        7: "DEFER",              # Skip for now, re-queue (max 3x)
    }

    # Action properties for simulation
    ACTION_PROPERTIES = {
        "ALLOW": {
            "processing_cost": 0.01,
            "delay": 0,
            "requires_human": False,
            "reversible": True,          # Can be undone later
            "user_impact": "none",
        },
        "FLAG_TICKET": {
            "processing_cost": 0.1,
            "delay": 300,                 # Seconds until human review
            "requires_human": True,
            "reversible": True,
            "user_impact": "none",
        },
        "FLAG_AUTO": {
            "processing_cost": 0.05,
            "delay": 60,
            "requires_human": False,
            "reversible": True,
            "user_impact": "none",
        },
        "REMOVE_IMMEDIATE": {
            "processing_cost": 0.02,
            "delay": 0,
            "requires_human": False,
            "reversible": True,
            "user_impact": "author_warning",
        },
        "REMOVE_WARN": {
            "processing_cost": 0.02,
            "delay": 0,
            "requires_human": False,
            "reversible": True,
            "user_impact": "author_ban_strike",
        },
        "ESCALATE_HIGH": {
            "processing_cost": 0.5,
            "delay": 60,
            "requires_human": True,
            "reversible": True,
            "user_impact": "none",
        },
        "ESCALATE_MISINFO": {
            "processing_cost": 0.5,
            "delay": 180,
            "requires_human": True,
            "reversible": True,
            "user_impact": "none",
        },
        "DEFER": {
            "processing_cost": 0.001,
            "delay": 0,
            "requires_human": False,
            "reversible": False,          # May auto-escalate if deferred too long
            "user_impact": "none",
        },
    }
```

### 4.2 Action Constraints

```python
"""
Action constraints simulate real-world business rules:
"""

CONSTRAINTS = {
    # Rate limits
    "max_removes_per_hour": 1000,
    "max_escalates_per_hour": 100,
    "max_deferrals_per_episode": 10,

    # Conditional restrictions
    "verified_authors_immune": False,      # Can always appeal
    "misinfo_escalation_only_for_misinfo": True,

    # Budget constraints
    "hourly_human_review_budget": 500,
    "cost_per_escalation": 1.0,
    "cost_per_remove": 0.1,
}
```

---

## 5. State Transitions

### 5.1 Transition Dynamics

```python
"""
State transitions are stochastic and realistic:
"""

def transition(state: WorldState, action: int) -> tuple[WorldState, Observation, dict]:
    next_state = copy.deepcopy(state)

    # 1. Process current action
    reward = compute_reward(state, action)
    is_terminal = False

    # 2. Advance the world clock
    step_duration = get_step_duration(action)  # seconds simulated

    # 3. New reports arrive for posts in system
    next_state = arrive_reports(state, next_state, step_duration)

    # 4. Deferred posts auto-expire
    next_state = process_deferred_posts(next_state)

    # 5. Queue management
    next_state = manage_queue(next_state)

    # 6. Ground truth revelation (delayed)
    if random() < 0.001:  # Rare immediate reveals
        next_state = reveal_ground_truth(next_state)

    # 7. User behavior response to actions
    next_state = user_behavior_response(state, action, next_state)

    # 8. Check terminal conditions
    is_terminal = check_terminal(next_state)

    # 9. Build observation
    obs = build_observation(next_state)

    # 10. Info dict
    info = {
        "action_taken": ACTION_SPACE.ACTIONS[action],
        "reward_breakdown": reward,
        "posts_processed": state.current_post is not None,
        "ground_truth_awaiting": get_awaiting_ground_truth(next_state),
        "system_metrics": get_system_metrics(next_state),
    }

    return next_state, obs, reward, is_terminal, info
```

### 5.2 Report Arrival Model

```python
"""
Reports arrive stochastically based on:
- Post toxicity level (higher toxicity → more reports)
- Time since posting (peak at 1-2 hours)
- Trending topic multiplier
- Whether post was already actioned
"""

def arrive_reports(state, next_state, dt_seconds):
    for post_id, post_reports in next_state.active_reports.items():
        if post_id in next_state.revealed_ground_truth:
            continue  # No more reports after final label

        # Base arrival rate
        base_rate = get_base_report_arrival_rate(post_reports[-1])

        # Toxicity multiplier (0.5 for safe, 2.0 for toxic)
        toxicity_mult = TOXICITY_MULTIPLIER[post_reports.ground_truth]

        # Time decay (reports peak then fade)
        time_factor = get_time_decay_factor(post.age_hours)

        # Action suppression (removal stops reports)
        if was_removed(post_id):
            continue

        # Trending boost
        trending = next_state.trending_topic_factor

        # Stochastic arrival
        expected_reports = base_rate * toxicity_mult * time_factor * trending * (dt_seconds / 60)
        actual_reports = np.random.poisson(expected_reports)

        for _ in range(actual_reports):
            report = generate_realistic_report(post_id)
            next_state.active_reports[post_id].append(report)

    return next_state
```

### 5.3 Ground Truth Revelation

```python
"""
Ground truth (final label) is revealed with delay:
- Toxic content: avg 4h, std 2h (reported faster)
- Spam: avg 2h, std 1h (reported faster due to spam filters)
- Safe content: avg 72h, std 24h (no reports, never fully resolved)

Resolution probability per step (simulates batch labeling):
"""

def reveal_ground_truth(state, post_id):
    post = state.posts[post_id]
    label = post.ground_truth_label

    delay_factor = {
        "toxic": 0.05,      # High chance of fast resolution
        "spam": 0.10,      # Spam filters accelerate
        "safe": 0.001,     # Rarely definitively labeled
    }[label]

    if random() < delay_factor:
        state.revealed_ground_truth[post_id] = label

        # Reveal final report count too
        state.posts[post_id].final_report_count = \
            state.posts[post_id].initial_report_count + \
            sum(len(r) for r in state.active_reports[post_id])

    return state
```

---

## 6. Reward Function

### 6.1 Multi-Component Reward

```python
def compute_reward(state: WorldState, action: int) -> float:
    """
    Reward composed of multiple components:
    R = R_accuracy + R_latency + R_cost + R_user_impact
    """
    reward = 0.0

    # Post being actioned
    post = state.active_review_post
    if post is None:
        return -0.1  # Wasted step penalty

    # Ground truth not yet known - partial reward based on report signals
    if post.post_id not in state.revealed_ground_truth:
        reward = compute_signal_based_reward(state, action, post)
    else:
        # Full accuracy reward
        reward = compute_accuracy_reward(state, action, post)

    # Latency penalty (delayed decisions cost more)
    reward -= LATENCY_PENALTY * state.review_duration

    # Cost efficiency
    reward -= ACTION_COST[action]

    return reward


def compute_accuracy_reward(state, action, post):
    """Full reward when ground truth is known."""
    gt = post.ground_truth_label
    action_type = get_action_category(action)

    # Accuracy matrix
    ACCURACY_REWARDS = {
        # (action_category, ground_truth): reward
        ("allow", "safe"):       +1.0,
        ("allow", "toxic"):      -1.0,   # False negative: missed toxicity
        ("allow", "spam"):       -0.8,   # Missed spam
        ("remove", "safe"):      -1.0,   # False positive: wrongly removed
        ("remove", "toxic"):     +0.8,   # Correct removal
        ("remove", "spam"):      +0.7,   # Correct spam removal
        ("flag", "safe"):        -0.3,   # Unnecessary queue load
        ("flag", "toxic"):       +0.5,   # Correct flag
        ("flag", "spam"):        +0.4,   # Correct flag
        ("escalate", "safe"):    -0.5,   # Waste of specialist time
        ("escalate", "toxic"):   +0.6,   # Correct escalation
        ("escalate", "spam"):    +0.5,
        ("defer", "safe"):       -0.1,   # Small cost for deferring
        ("defer", "toxic"):      -0.5,   # Bigger cost for deferring toxic
        ("defer", "spam"):       -0.4,
    }

    return ACCURACY_REWARDS.get((action_type, gt), 0.0)


def compute_signal_based_reward(state, action, post):
    """
    Reward based on report signals when ground truth unknown.
    This shapes good behavior during the waiting period.
    """
    signal_score = compute_toxicity_signal(state, post)

    # If high signal but actioned lightly, small penalty
    if signal_score > 0.8 and action in ["ALLOW", "DEFER"]:
        return -0.2 * signal_score

    # If low signal but actioned heavily, small penalty
    if signal_score < 0.2 and action in ["REMOVE_IMMEDIATE", "ESCALATE_HIGH"]:
        return -0.2 * (1 - signal_score)

    # Neutral/small positive for proportional response
    return 0.0
```

### 6.2 Reward Parameters

```python
REWARD_PARAMS = {
    # Accuracy rewards
    "true_positive_remove": 1.0,
    "true_positive_flag": 0.6,
    "false_positive": -1.0,
    "false_negative": -1.0,
    "true_negative_allow": 0.1,         # Small reward for allowing safe

    # Latency (per step in review)
    "latency_penalty_per_step": 0.01,

    # Cost (per action)
    "cost_allow": 0.01,
    "cost_flag": 0.10,
    "cost_remove": 0.05,
    "cost_escalate": 0.50,
    "cost_defer": 0.02,

    # Cascading effects
    "author_ban_repeat_toxic_penalty": -0.5,  # Applied to future steps
    "platform_trust_damage_toxic_leak": -0.2, # Applied to future steps
}
```

---

## 7. Episode Structure

### 7.1 Episode Definition

```python
"""
Episode = One moderation shift (e.g., 1 hour of simulated time)

Episode can end due to:
1. Time limit reached (shift ends)
2. Queue emptied AND no pending ground truth resolutions
3. Catastrophic failure (optional constraint)
"""

class EpisodeConfig:
    duration_steps: int = 3600          # 3600 steps × 1 second = 1 hour
    max_posts_per_episode: int = 500    # Cap on posts to process

    # Queue injection
    posts_per_minute: float = 10.0       # Average arrival rate
    posts_arrival_std: float = 3.0      # Standard deviation

    # Initial conditions
    initial_queue_depth: int = 20       # Posts already waiting
    initial_budget: float = 100.0       # Human review budget

    # Difficulty scaling
    toxic_ratio: float = 0.15           # 15% toxic
    spam_ratio: float = 0.20            # 20% spam
    safe_ratio: float = 0.65            # 65% safe
```

### 7.2 Episode Phases

```python
"""
Each episode has 4 phases with different characteristics:
"""

PHASES = {
    "morning_rush": {
        "start_minute": 0,
        "end_minute": 15,
        "arrival_multiplier": 1.5,
        "toxic_ratio_multiplier": 1.2,
        "trending_factor": 1.1,
    },
    "midday_steady": {
        "start_minute": 15,
        "end_minute": 45,
        "arrival_multiplier": 1.0,
        "toxic_ratio_multiplier": 1.0,
        "trending_factor": 1.0,
    },
    "evening_spike": {
        "start_minute": 45,
        "end_minute": 55,
        "arrival_multiplier": 2.0,
        "toxic_ratio_multiplier": 1.5,
        "trending_factor": 1.5,
    },
    "late_evening": {
        "start_minute": 55,
        "end_minute": 60,
        "arrival_multiplier": 0.7,
        "toxic_ratio_multiplier": 0.8,
        "trending_factor": 0.9,
    },
}
```

### 7.3 Terminal Conditions

```python
def is_terminal(state: WorldState) -> bool:
    """Episode ends when any of these are true."""
    return any([
        # Time limit
        state.elapsed_steps >= EPISODE_CONFIG.duration_steps,

        # Post limit
        state.total_posts_processed >= EPISODE_CONFIG.max_posts_per_episode,

        # Queue drain (no posts left, no pending resolutions)
        (len(state.pending_posts) == 0 and
         len(state.active_review_post) == 0 and
         len(state.awaiting_resolution) == 0),

        # Budget exhaustion
        state.budget_remaining <= 0,
    ])
```

---

## 8. Environment API

### 8.1 Gymnasium Interface

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class ContentGuardEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, config: dict | None = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG

        # Spaces
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(78,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(8)

        # Internal state
        self.state = None
        self.episode_step = 0
        self._action_history = []

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.state = initialize_world_state(self.config, seed)
        self.episode_step = 0
        self._action_history = []

        obs = build_observation(self.state)
        info = self._get_info()

        return obs, info

    def step(self, action):
        """Execute action and return (obs, reward, terminal, truncated, info)."""
        self._action_history.append(action)

        # Transition
        self.state, obs, reward, terminal, info = transition(
            self.state, action
        )

        self.episode_step += 1

        # Truncated = violated constraints (optional safety)
        truncated = self._check_constraints()

        return obs, reward, terminal, truncated, info

    def render(self, mode="human"):
        """Render current state."""
        if mode == "human":
            self._render_human()
        elif mode == "rgb_array":
            return self._render_rgb()

    def close(self):
        """Cleanup."""
        pass
```

### 8.2 Configuration Options

```python
DEFAULT_CONFIG = {
    # Environment mode
    "mode": "production",          # "production" | "research"

    # Observation settings
    "partial_observability": True,  # If True, hide some delayed features
    "embedding_model": "tfidf",    # "tfidf" | "transformer" | "random"

    # Simulation parameters
    "time_dilation": 1.0,           # 1.0 = real-time, 10.0 = 10x faster
    "arrival_rate_mean": 10.0,      # Posts per minute
    "arrival_rate_std": 3.0,

    # Content distribution
    "toxic_ratio": 0.15,
    "spam_ratio": 0.20,
    "safe_ratio": 0.65,

    # Reward weights
    "reward_weights": {
        "accuracy": 1.0,
        "latency": 0.1,
        "cost": 0.05,
    },

    # Constraints
    "max_actions_per_hour": 5000,
    "max_escalations_per_hour": 200,
    "human_review_budget": 1000,

    # Episode
    "episode_duration_seconds": 3600,
    "max_posts_per_episode": 500,
}
```

---

## 9. Evaluation Metrics

### 9.1 Performance Metrics

```python
"""
Metrics computed at episode end for agent evaluation:
"""

EVALUATION_METRICS = {
    # Accuracy metrics
    "accuracy": "Proportion of correct decisions",
    "precision": "True positives / (True positives + False positives)",
    "recall": "True positives / (True positives + False negatives)",
    "f1_score": "Harmonic mean of precision and recall",

    # Toxic-specific metrics
    "toxic_precision": "Precision on toxic posts specifically",
    "toxic_recall": "Recall on toxic posts specifically",
    "toxic_f1": "F1 on toxic posts specifically",

    # False positive/negative rates
    "false_positive_rate": "Safe posts incorrectly removed",
    "false_negative_rate": "Toxic posts incorrectly allowed",
    "missed_toxic_rate": "Toxic posts allowed (FN / total toxic)",

    # Efficiency metrics
    "avg_latency": "Average time in queue per post",
    "p95_latency": "95th percentile queue time",
    "p99_latency": "99th percentile queue time",
    "queue_overflow_rate": "Posts lost due to queue overflow",
    "throughput": "Posts processed per simulated minute",

    # Cost metrics
    "total_cost": "Sum of action costs",
    "escalation_rate": "Proportion of posts escalated",
    "human_review_load": "Human reviewer minutes required",

    # Fairness metrics
    "disparate_impact_by_author_age": "FP/FN rate by author account age",
    "disparate_impact_by_verified": "FP/FN rate by verified status",
    "disparate_impact_by_language": "FP/FN rate by language",
}
```

### 9.2 Baseline Agents for Comparison

```python
BASELINE_AGENTS = {
    "random": RandomAgent,              # Random action selection
    "majority_vote": MajorityVoteAgent,  # Allow if reports < threshold
    "report_threshold": ThresholdAgent, # Remove if report_count > threshold
    "trust_weighted": TrustWeightedAgent,  # Weighted by reporter trust
    "rule_based": RuleBasedAgent,        # Hard-coded rules (production baseline)
    "oracle": OracleAgent,               # Knows ground truth (upper bound)
}
```

---

## 10. Dataset Generation

### 10.1 Synthetic Post Generation

```python
"""
Posts generated using template-based approach with realistic variation:
"""

POST_TEMPLATES = {
    "toxic": [
        "You're such a {insult}! Everyone knows you're {falsehood}",
        "I hope you {violent_act}. You {protected_group} are the worst",
        "{slur} alert: {target}",
        # ... 100+ templates
    ],
    "spam": [
        "Make ${amount} from home! Click here: {url}",
        "FREE {product}!!! Limited time offer!!! {url}",
        # ... 50+ templates
    ],
    "safe": [
        "Just finished watching {movie}. It was {sentiment}",
        "Happy birthday to my {relationship}! Love you {amount}",
        "Does anyone know how to fix {common_problem}?",
        # ... 200+ templates
    ],
}

# Real-world text statistics
TEXT_DISTRIBUTIONS = {
    "text_length": {"mean": 120, "std": 80, "min": 10, "max": 500},
    "emoji_probability": {"toxic": 0.3, "spam": 0.5, "safe": 0.4},
    "url_probability": {"toxic": 0.1, "spam": 0.8, "safe": 0.2},
    "hashtag_probability": {"toxic": 0.4, "spam": 0.6, "safe": 0.3},
}

# Author distributions
AUTHOR_DISTRIBUTIONS = {
    "account_age_days": {
        "toxic": {"mean": 45, "std": 60, "median": 14},
        "spam": {"mean": 30, "std": 45, "median": 7},
        "safe": {"mean": 365, "std": 400, "median": 180},
    },
    "post_count": {
        "toxic": {"mean": 50, "std": 100, "median": 15},
        "spam": {"mean": 20, "std": 50, "median": 5},
        "safe": {"mean": 500, "std": 1000, "median": 200},
    },
    "ban_history": {
        "toxic": {"mean": 1.5, "std": 2.0},
        "spam": {"mean": 0.5, "std": 1.0},
        "safe": {"mean": 0.0, "std": 0.1},
    },
}
```

### 10.2 Real-World Dataset Integration

```python
"""
Can optionally load real datasets:
- Jigsaw Unintended Bias in Toxicity (kaggle)
- HateXplain (huggingface)
- CivilComments (kaggle)
- Twitter hate speech datasets

Environment normalizes these to Post format:
"""

def load_real_dataset(name: str) -> list[Post]:
    if name == "jigsaw":
        return load_jigsaw_toxicity_dataset()
    elif name == "hatexplain":
        return load_hatexplain_dataset()
    elif name == "civilcomments":
        return load_civil_comments_dataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")
```

---

## 11. Implementation Notes

### 11.1 Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Partial observability | Production systems never have ground truth immediately |
| Delayed ground truth | Simulates real labeling latency |
| Stochastic transitions | Real world is not deterministic |
| Multi-component reward | Real moderation has multiple objectives |
| Queue management | Simulates real backpressure |
| Budget constraints | Human review is expensive/limited |
| User behavior response | Actions affect future report rates |

### 11.2 Limitations

1. **Synthetic text embeddings**: Real production would use actual ML models
2. **Simplified report dynamics**: Real report arrival is more complex
3. **No user appeals**: Appeals process adds significant complexity
4. **No coordinated campaigns**: Sophisticated adversarial evasions not modeled
5. **Single-platform**: Doesn't capture cross-platform dynamics

### 11.3 Extension Points

- Add multi-language support
- Add coordinated attack detection
- Add temporal pattern learning (same user posts over time)
- Add platform network effects (virality, cascading reports)
- Add adversarial robustness testing
- Add multi-action batch processing

---

## 12. Quick Start

```python
import gymnasium as gym
from contentguard import ContentGuardEnv

# Create environment
env = gym.make("ContentGuard-v1", config={"mode": "production"})

# Run episode
obs, info = env.reset()
done = False
total_reward = 0

while not done:
    action = your_agent(obs)  # Your RL agent
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated

# Evaluate
metrics = info.get("episode_metrics", {})
print(f"Episode complete. F1: {metrics['f1_score']:.3f}, "
      f"Reward: {total_reward:.2f}")
```

---

*Specification version 1.0.0 - ContentGuard RL Environment*
