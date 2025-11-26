"""Math utilities for Hugging Face community score calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from config.scoring_config import HUGGING_FACE_SCORE_PARAMS


@dataclass(frozen=True)
class HFScoreTelemetry:
    """Minimal telemetry required to compute the HF community score."""

    downloads: int
    likes: int
    created_at: datetime


@dataclass(frozen=True)
class HFScoreResult:
    """Represents the score output plus helpful intermediate metrics."""

    hf_score: float
    download_score: float
    likes_score: float
    age_score: float
    age_months: float
    age_weeks: int


def calculate_download_score(downloads: int) -> float:
    """Calculate the downloads contribution using shared config."""

    params = HUGGING_FACE_SCORE_PARAMS["downloads"]
    if downloads < params["min_downloads"]:
        return 0.0

    log_val = math.log(downloads) / math.log(params["log_base"])
    score = params["coefficient"] * log_val + params["intercept"]
    return max(0.0, min(params["max_points"], score))


def calculate_likes_score(likes: int) -> float:
    """Calculate the likes contribution using shared config."""

    params = HUGGING_FACE_SCORE_PARAMS["likes"]
    if likes < params["min_likes"]:
        return 0.0

    log_val = math.log(likes) / math.log(params["log_base"])
    score = params["coefficient"] * log_val + params["intercept"]
    return max(0.0, min(params["max_points"], score))


def calculate_age_score(age_months: float) -> float:
    """Calculate the maturity contribution using shared config."""

    params = HUGGING_FACE_SCORE_PARAMS["age_months"]

    if 0 <= age_months < params["tier1_months"]:
        score = params["tier1_slope"] * age_months
    elif params["tier1_months"] <= age_months < params["tier2_months"]:
        score = params["tier2_base_points"] + params["tier2_slope"] * (
            age_months - params["tier1_months"]
        )
    elif params["tier2_months"] <= age_months <= params["tier3_months"]:
        score = params["tier3_base_points"] + params["tier3_slope"] * (
            age_months - params["tier2_months"]
        )
    else:
        score = params["stable_points"]

    return max(0.0, min(params["max_points"], score))


def compute_score(
    telemetry: HFScoreTelemetry, *, now: Optional[datetime] = None
) -> HFScoreResult:
    """Compute the HF community score using telemetry from any source."""

    if now is None:
        now = datetime.now(timezone.utc)

    if telemetry.created_at.tzinfo is None:
        created_at = telemetry.created_at.replace(tzinfo=timezone.utc)
    else:
        created_at = telemetry.created_at.astimezone(timezone.utc)

    age_delta = now - created_at
    age_weeks = age_delta.days // 7
    age_months = age_delta.days / 30.437

    download_score = calculate_download_score(max(telemetry.downloads, 0))
    likes_score = calculate_likes_score(max(telemetry.likes, 0))
    age_score = calculate_age_score(max(age_months, 0.0))

    total_score = round(download_score + likes_score + age_score, 2)

    return HFScoreResult(
        hf_score=total_score,
        download_score=round(download_score, 4),
        likes_score=round(likes_score, 4),
        age_score=round(age_score, 4),
        age_months=round(age_months, 4),
        age_weeks=age_weeks,
    )


__all__ = [
    "HFScoreTelemetry",
    "HFScoreResult",
    "calculate_download_score",
    "calculate_likes_score",
    "calculate_age_score",
    "compute_score",
]
