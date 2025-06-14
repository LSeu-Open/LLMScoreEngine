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

This module provides a function to calculate the community score for a model
based on its downloads, likes, and age.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from huggingface_hub import model_info
from datetime import datetime, timezone
import math
import argparse

import sys
from pathlib import Path

# Add project root to sys.path for absolute imports, making the script runnable.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.scoring_config import HUGGING_FACE_SCORE_PARAMS

# ------------------------------------------------------------------------------------------------
# Hugging Face Community Score Functions
# ------------------------------------------------------------------------------------------------

def _calculate_download_score(downloads: int) -> float:
    """Calculates the download score based on the formula in scoring_framework.md."""
    params = HUGGING_FACE_SCORE_PARAMS['downloads']
    if downloads < params['min_downloads']:
        return 0.0
    
    log_val = math.log(downloads) / math.log(params['log_base'])
    score = params['coefficient'] * log_val + params['intercept']
    return max(0.0, min(params['max_points'], score))

def _calculate_likes_score(likes: int) -> float:
    """Calculates the likes score based on the formula in scoring_framework.md."""
    params = HUGGING_FACE_SCORE_PARAMS['likes']
    if likes < params['min_likes']:
        return 0.0
        
    log_val = math.log(likes) / math.log(params['log_base'])
    score = params['coefficient'] * log_val + params['intercept']
    return max(0.0, min(params['max_points'], score))

def _calculate_age_score(age_months: float) -> float:
    """Calculates the age/maturity score based on the formula in scoring_framework.md."""
    params = HUGGING_FACE_SCORE_PARAMS['age_months']
    
    if 0 <= age_months < params['tier1_months']:
        score = params['tier1_slope'] * age_months
    elif params['tier1_months'] <= age_months < params['tier2_months']:
        score = params['tier2_base_points'] + params['tier2_slope'] * (age_months - params['tier1_months'])
    elif params['tier2_months'] <= age_months <= params['tier3_months']:
        score = params['tier3_base_points'] + params['tier3_slope'] * (age_months - params['tier2_months'])
    else:  # age_months > params['tier3_months']
        score = params['stable_points']
        
    return max(0.0, min(params['max_points'], score))

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
    """Computes the total Hugging Face Community Score based on documentation."""
    download_score = _calculate_download_score(model_info["downloads in last 30 days"])
    likes_score = _calculate_likes_score(model_info["total likes"])
    age_score = _calculate_age_score(model_info["age in months"])
    
    total_score = download_score + likes_score + age_score
    return round(total_score, 2)

def extract_model_info(model_name: str) -> dict:
    """Extract all information for a model and return as a dictionary."""
    downloads = get_model_downloads(model_name)
    likes = get_model_likes(model_name)
    age_weeks, age_months = get_model_age(model_name)
    
    info = {
        "model_name": model_name,
        "downloads in last 30 days": downloads,
        "total likes": likes,
        "age in weeks": age_weeks,
        "age in months": age_months
    }
    
    # Add community score
    info["community_score"] = compute_hf_score(info)
    
    return info

# ------------------------------------------------------------------------------------------------
# Main function
# ------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get Hugging Face model community score and metrics.")
    parser.add_argument("model_name", help="Name of the Hugging Face model (e.g., microsoft/Phi-4-mini-reasoning)")
    args = parser.parse_args()

    # Example usage
    model_name = args.model_name # Change this to the model you want to score
    
    # Get all metrics at once
    info = extract_model_info(model_name)
    
    # Print score prominently
    print(f"\nHF COMMUNITY SCORE: {info['community_score']}/10")
    
    print("\nDetailed metrics:")
    for key, value in info.items():
        if key != "community_score":  # Skip community_score as we already displayed it
            print(f"{key}: {value}")



