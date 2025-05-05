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
Scoring configuration for the model scoring system.

This module contains all the configurable parameters for the scoring system,
including weights, thresholds, and bounds for different scoring components.
"""

# Maximum points for each category
SCORE_WEIGHTS = {
    'entity_benchmarks': 25,  # Maximum points for entity benchmarks
    'dev_benchmarks': 35,     # Maximum points for dev benchmarks
    'community_score': 20,    # Maximum points for community score
    'technical_score': 20     # Maximum points for technical score
}

# Thresholds for technical scoring
SCORE_THRESHOLDS = {
    'price': {
        'excellent': 1,    # < 1 USD per million tokens
        'good': 3,         # < 3 USD
        'fair': 5,         # < 5 USD
        'poor': 10,        # < 10 USD
        'bad': 20,         # < 20 USD
        'very_bad': 40,    # < 40 USD
        'worst': 80        # < 80 USD
    },
    'context_window': {
        'excellent': 200000,  # > 200k tokens
        'good': 128000,       # > 128k tokens
        'fair': 64000,        # > 64k tokens
        'poor': 32000,        # > 32k tokens
        'bad': 16000,         # > 16k tokens
        'very_bad': 8000,     # > 8k tokens
        'worst': 4000         # > 4k tokens
    },
    'size_perf_ratio': {
        'excellent': 90,  # > 90%
        'good': 80,       # > 80%
        'fair': 70,       # > 70%
        'poor': 60,       # > 60%
        'bad': 50,        # > 50%
        'very_bad': 40,   # > 40%
        'worst': 30       # > 30%
    }
}

# Weights for different benchmarks
BENCHMARK_WEIGHTS = {
    'entity_benchmarks': {
        'artificial_analysis': 25,
        'live_code_bench': 25,
        'big_code_models': 25,
        'open_llm': 25
    },
    'dev_benchmarks': {
        'MMLU': 3.0,
        'MMLU Pro': 8.0,
        'BigBench': 3.0,
        'DROP': 7.0,
        'HellaSwag': 7.0,
        'GPQA': 2.0,
        'ARC-C': 2.0,
        'LiveBench': 1.5,
        'LatestEval': 1.5,
        'AlignBench': 4.0,
        'Wild Bench': 4.0,
        'MT-bench': 4.0,
        'IFEval': 4.0,
        'Arena-Hard': 4.5,
        'TruthfulQA': 4.5,
        'MATH': 4.0,
        'GSM-8K': 4.0,
        'MGSM': 7.0,
        'HumanEval': 3.0,
        'HumanEval Plus': 3.0,
        'MBPP': 3.0,
        'MBPP Plus': 3.0,
        'SWE-bench': 2.0,
        'API-Bank': 2.0,
        'BFCL': 5.0,
        'Gorilla Benchmark': 2.0,
        'Nexus': 2.0
    }
}

# Bounds for community score calculation
COMMUNITY_SCORE_BOUNDS = {
    'min_elo': 1000,  # Minimum expected ELO rating
    'max_elo': 1402   # Maximum expected ELO rating
} 