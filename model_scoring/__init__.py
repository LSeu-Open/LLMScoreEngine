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
Model Scoring System

A comprehensive system for evaluating and scoring large language models
based on multiple criteria including benchmark performance, community
ratings, and technical specifications.

This is the Alpha v0.3 of the scoring system.
"""

from .scoring.models_scoring import ModelScorer
from .run_scoring import batch_process_models
from .data.loaders import load_model_data
from .data.validators import validate_model_data

__version__ = "0.3.2"
__author__ = "LSeu-Open"

__all__ = [
    'ModelScorer',
    'batch_process_models',
    'load_model_data',
    'validate_model_data'
] 