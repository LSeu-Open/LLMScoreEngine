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
Tests for the Hugging Face score calculation.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta, timezone

from model_scoring.scoring import hf_score
from model_scoring.scoring import hf_score_math

from config.scoring_config import HUGGING_FACE_SCORE_PARAMS

# Forcing a re-evaluation due to a potential caching issue.

# Test data for mocking
MOCK_MODEL_INFO = {
    "downloads": 50000,
    "likes": 1500,
    "created_at": datetime.now(timezone.utc) - timedelta(days=180) # Approx 6 months old
}

@pytest.fixture
def mock_hf_api():
    """Fixture to mock the huggingface_hub.model_info call."""
    with patch('model_scoring.scoring.hf_score.model_info') as mock_model_info:
        mock_instance = MagicMock()
        mock_instance.downloads = MOCK_MODEL_INFO['downloads']
        mock_instance.likes = MOCK_MODEL_INFO['likes']
        mock_instance.created_at = MOCK_MODEL_INFO['created_at']
        mock_model_info.return_value = mock_instance
        yield mock_model_info

def test_calculate_download_score():
    """Tests the download score calculation with various inputs."""
    params = HUGGING_FACE_SCORE_PARAMS['downloads']
    
    # Test case 1: Below minimum downloads
    score1 = hf_score._calculate_download_score(params['min_downloads'] - 1)
    print(f"\n[Download Score] Below min ({params['min_downloads'] - 1} downloads): {score1}")
    assert score1 == 0.0
    
    # Test case 2: A specific known value
    # Using the formula: 0.2007 * log2(20000) + -0.6667 = 2.22
    score2 = hf_score._calculate_download_score(20000)
    print(f"[Download Score] 20k downloads: {score2:.4f} (rounded: {round(score2, 2)})")
    assert round(score2, 2) == 2.20
    
    # Test case 3: A value that would exceed max_points without clamping
    # 10M downloads should definitely hit the max
    score3 = hf_score._calculate_download_score(10_000_000)
    print(f"[Download Score] 10M downloads (clamped): {score3}")
    assert score3 == params['max_points']

def test_calculate_likes_score():
    """Tests the likes score calculation."""
    params = HUGGING_FACE_SCORE_PARAMS['likes']
    
    # Test case 1: Below minimum likes
    score1 = hf_score._calculate_likes_score(params['min_likes'] - 1)
    print(f"\n[Likes Score] Below min ({params['min_likes'] - 1} likes): {score1}")
    assert score1 == 0.0
    
    # Test case 2: A specific known value
    # Using the formula: 0.477 * log2(1000) + -0.756 = 3.99
    score2 = hf_score._calculate_likes_score(1000)
    print(f"[Likes Score] 1k likes: {score2:.4f} (rounded: {round(score2, 2)})")
    assert round(score2, 2) == 4.00

    # Test case 3: High value to test clamping at max_points
    score3 = hf_score._calculate_likes_score(50000)
    print(f"[Likes Score] 50k likes (clamped): {score3}")
    assert score3 == params['max_points']

def test_calculate_age_score():
    """Tests the age score calculation across different tiers."""
    params = HUGGING_FACE_SCORE_PARAMS['age_months']

    # Tier 1: 0 to 1 month
    age_tier1 = 0.5
    score1 = hf_score._calculate_age_score(age_tier1)
    print(f"\n[Age Score] Tier 1 ({age_tier1} months): {score1:.4f}")
    assert round(score1, 2) == round(params['tier1_slope'] * age_tier1, 2)
    
    # Tier 2: 1 to 3 months
    age_tier2 = 2.0
    score2 = hf_score._calculate_age_score(age_tier2)
    expected_score_t2 = params['tier2_base_points'] + params['tier2_slope'] * (age_tier2 - params['tier1_months'])
    print(f"[Age Score] Tier 2 ({age_tier2} months): {score2:.4f}")
    assert round(score2, 2) == round(expected_score_t2, 2)
    
    # Tier 3: 3 to 12 months
    age_tier3 = 7.0
    score3 = hf_score._calculate_age_score(age_tier3)
    expected_score_t3 = params['tier3_base_points'] + params['tier3_slope'] * (age_tier3 - params['tier2_months'])
    print(f"[Age Score] Tier 3 ({age_tier3} months): {score3:.4f}")
    assert round(score3, 2) == round(expected_score_t3, 2)
    
    # Stable: > 12 months
    age_stable = 15
    score4 = hf_score._calculate_age_score(age_stable)
    print(f"[Age Score] Stable ({age_stable} months): {score4}")
    assert score4 == params['stable_points']
    
    # Test clamping at max_points (if any calculation could exceed it)
    score5 = hf_score._calculate_age_score(100)
    print(f"[Age Score] Clamped at max ({100} months): {score5}")
    assert score5 <= params['max_points']

def test_get_model_age(mock_hf_api):
    """Tests the model age retrieval and calculation."""
    age_weeks, age_months = hf_score.get_model_age("mock/model")
    print(f"\n[Model Age] Calculated age: {age_months:.2f} months ({age_weeks} weeks)")
    # Should be around 6 months
    assert 5.9 < age_months < 6.1

def test_compute_hf_score():
    """Tests the main computation function with sample data."""
    # Create a sample model info dict based on our mock data
    age_in_days = (datetime.now(timezone.utc) - MOCK_MODEL_INFO['created_at']).days
    age_in_months = age_in_days / 30.437

    model_data = {
        "downloads in last 30 days": MOCK_MODEL_INFO['downloads'],
        "total likes": MOCK_MODEL_INFO['likes'],
        "age in months": age_in_months
    }
    
    # Calculate expected scores individually
    download_score = hf_score._calculate_download_score(model_data["downloads in last 30 days"])
    likes_score = hf_score._calculate_likes_score(model_data["total likes"])
    age_score = hf_score._calculate_age_score(model_data["age in months"])
    
    expected_total = round(download_score + likes_score + age_score, 2)
    actual_total = hf_score.compute_hf_score(model_data)
    
    assert actual_total == expected_total

def test_compute_score_math():
    """Ensure math helper produces deterministic telemetry results."""
    telemetry = hf_score_math.HFScoreTelemetry(
        downloads=MOCK_MODEL_INFO['downloads'],
        likes=MOCK_MODEL_INFO['likes'],
        created_at=MOCK_MODEL_INFO['created_at'],
    )

    result = hf_score_math.compute_score(telemetry, now=MOCK_MODEL_INFO['created_at'] + timedelta(days=180))

    assert result.hf_score == hf_score.compute_hf_score({
        "downloads in last 30 days": telemetry.downloads,
        "total likes": telemetry.likes,
        "age in months": result.age_months,
    })
    assert result.download_score == round(result.download_score, 4)
    assert result.likes_score == round(result.likes_score, 4)
    assert result.age_score == round(result.age_score, 4)

def test_extract_model_info(mock_hf_api):
    """Tests the full data extraction and scoring process, mocking the API call."""
    model_name = "mock/model"
    info = hf_score.extract_model_info(model_name)
    
    print(f"\n[Extracted Info & Score Breakdown] for {model_name}:")
    print(f"  - Downloads: {info['downloads in last 30 days']}")
    print(f"  - Likes: {info['total likes']}")
    print(f"  - Age (months): {info['age in months']:.2f}")
    
    download_score = hf_score._calculate_download_score(info["downloads in last 30 days"])
    likes_score = hf_score._calculate_likes_score(info["total likes"])
    age_score = hf_score._calculate_age_score(info["age in months"])
    
    print("  ---")
    print(f"  - Download Score: {download_score:.4f}")
    print(f"  - Likes Score:    {likes_score:.4f}")
    print(f"  - Age Score:      {age_score:.4f}")
    print("  ---")
    print(f"  - Final Community Score: {info['community_score']}")

    assert info['model_name'] == model_name
    assert info['downloads in last 30 days'] == MOCK_MODEL_INFO['downloads']
    assert info['total likes'] == MOCK_MODEL_INFO['likes']
    assert 'community_score' in info
    
    # Re-calculate score to ensure it's consistent
    recalculated_score = hf_score.compute_hf_score(info)
    assert info['community_score'] == recalculated_score 