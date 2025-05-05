"""
Constants for the model scoring system.

This module defines the constants used throughout the scoring system,
including score bounds, required sections for validation, and directory paths.
"""

# Score scale and bounds
SCORE_SCALE = 100

SCORE_BOUNDS = {
    "MIN": 0,
    "MAX": 100
}

# Community score bounds (based on the community score of the best models 20 feb 2025)
COMMUNITY_SCORE_BOUNDS = {
    "MIN": 1000,
    "MAX": 1500
}

# Directory and file constants
MODELS_DIR = "Models"
RESULTS_DIR = "Results"
LOG_FILE = "model_scoring.log"

# Required sections and fields that must be present in model JSON files for validation
REQUIRED_SECTIONS = {
    'entity_benchmarks': [
        'artificial_analysis',
        'live_code_bench',
        'big_code_models',
        'open_llm'
    ],
    'dev_benchmarks': [
        'MMLU', 'MMLU Pro', 'BigBench', 'DROP', 'HellaSwag', 'GPQA', 
        'ARC-C', 'LiveBench', 'LatestEval', 'AlignBench', 'Wild Bench',
        'MT-bench', 'IFEval', 'Arena-Hard', 'TruthfulQA', 'MATH',
        'GSM-8K', 'MGSM', 'HumanEval', 'HumanEval Plus', 'MBPP',
        'MBPP Plus', 'SWE-bench', 'API-Bank', 'BFCL',
        'Gorilla Benchmark', 'Nexus'
    ],
    'model_specs': ['price', 'context_window', 'param_count'],
    'community_score': None
} 