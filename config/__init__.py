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
Configuration module for the model scoring system.

This module provides centralized configuration management for the scoring system,
including scoring weights, thresholds, and other configurable parameters.
"""

from .scoring_config import (
    SCORE_WEIGHTS,
    BENCHMARK_WEIGHTS,
    COMMUNITY_SCORE_BOUNDS,
    HUGGING_FACE_SCORE_PARAMS,
    TECHNICAL_SCORE_PARAMS,
    MODEL_ARCHITECTURE_FACTORS,
    SCORE_SCALE,
    SCORE_BOUNDS
) 
