# ------------------------------------------------------------------------------------------------
# License
# ------------------------------------------------------------------------------------------------

# Copyright (c) 2025 LSeu-Open
# 
# This code is licensed under the MIT License.
# See LICENSE file in the root directory

# ------------------------------------------------------------------------------------------------
# Description
# ------------------------------------------------------------------------------------------------

"""
Hugging Face Community Score Calculation.

This module fetches telemetry from Hugging Face once, then delegates all math to
``hf_score_math`` so the formulas stay centralized.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from huggingface_hub import model_info
from datetime import datetime, timezone
import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from model_scoring.scoring.hf_score_math import (
    HFScoreResult,
    HFScoreTelemetry,
    calculate_age_score,
    calculate_download_score,
    calculate_likes_score,
    compute_score,
)

# Add project root to sys.path for absolute imports, making the script runnable.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ------------------------------------------------------------------------------------------------
# Hugging Face Community Score Functions
# ------------------------------------------------------------------------------------------------

_calculate_download_score = calculate_download_score
_calculate_likes_score = calculate_likes_score
_calculate_age_score = calculate_age_score

def get_model_downloads(model_name: str) -> int:
    """Get the last 30 days downloads for a model."""
    return model_info(model_name).downloads

def get_model_likes(model_name: str) -> int:
    """Get the number of likes for a model."""
    return model_info(model_name).likes

def get_model_age(model_name: str) -> tuple[int, float]:
    """Get the age of a model in weeks and months."""
    model = model_info(model_name)
    created_at = model.created_at
    now = datetime.now(timezone.utc)
    age_delta = now - created_at
    age_weeks = age_delta.days // 7
    age_months = age_delta.days / 30.437 # Average number of days in a month
    return age_weeks, age_months

def compute_hf_score(model_info: dict) -> float:
    """Compute the HF score for pre-extracted metrics (legacy helper)."""

    download_score = calculate_download_score(model_info["downloads in last 30 days"])
    likes_score = calculate_likes_score(model_info["total likes"])
    age_score = calculate_age_score(model_info["age in months"])

    total_score = download_score + likes_score + age_score
    return round(total_score, 2)


def extract_model_info(model_name: str) -> dict:
    """Fetch telemetry from Hugging Face and compute score in one pass."""

    model = model_info(model_name)
    telemetry = HFScoreTelemetry(
        downloads=model.downloads or 0,
        likes=model.likes or 0,
        created_at=model.created_at,
    )
    score_result: HFScoreResult = compute_score(telemetry)

    return {
        "model_name": model_name,
        "downloads in last 30 days": telemetry.downloads,
        "total likes": telemetry.likes,
        "age in weeks": score_result.age_weeks,
        "age in months": score_result.age_months,
        "community_score": score_result.hf_score,
        "download_score": score_result.download_score,
        "likes_score": score_result.likes_score,
        "age_score": score_result.age_score,
    }

# ------------------------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------------------------

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Get Hugging Face model community score and metrics."
    )
    parser.add_argument(
        "model_name",
        help="Name of the Hugging Face model (e.g., microsoft/Phi-4-mini-reasoning)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    info = extract_model_info(args.model_name)
    print(f"\nHF COMMUNITY SCORE: {info['community_score']}/10")
    print("\nDetailed metrics:")
    for key, value in info.items():
        if key != "community_score":
            print(f"{key}: {value}")
    return info


if __name__ == "__main__":
    main()



