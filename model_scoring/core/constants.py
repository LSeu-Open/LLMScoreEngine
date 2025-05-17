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

# Bounds for individual components of the community score
LM_SYS_ARENA_SCORE_BOUNDS = {
    "MIN": 1000,
    "MAX": 1500
}

HF_COMMUNITY_SCORE_BOUNDS = {
    "MIN": 0,
    "MAX": 10
}

# Directory and file constants
MODELS_DIR = "Models"
RESULTS_DIR = "Results"
LOG_FILE = "model_scoring.log"

# Required sections and fields that must be present in model JSON files for validation
REQUIRED_SECTIONS = {
    'entity_benchmarks': [
        'artificial_analysis',
        'OpenCompass',
        'LLM Explorer',
        'Livebench',
        'open_llm',
        'UGI Leaderboard',
        'big_code_bench',
        'EvalPlus Leaderboard',
        'Dubesord_LLM',
        'Open VLM',
    ],
    'dev_benchmarks': [
        'MMLU', 
        'MMLU Pro', 
        'BigBenchHard',
        'GPQA diamond', 
        'DROP', 
        'HellaSwag', 
        'Humanity\'s Last Exam',
        'ARC-C', 
        'Wild Bench',
        'MT-bench', 
        'IFEval', 
        'Arena-Hard',
        'MATH',
        'GSM-8K',
        'AIME',
        'HumanEval',
        'MBPP',
        'LiveCodeBench',
        'Aider Polyglot',
        'SWE-Bench',
        'SciCode',
        'MGSM',
        'MMMLU',
        'C-Eval or CMMLU',
        'AraMMLu',
        'LongBench',
        'RULER 128K',
        'RULER 32K',
        'MTOB',
        'BFCL',
        'AgentBench',
        'Gorilla Benchmark',
        'ToolBench',
        'MINT',
        'MMMU',
        'Mathvista',
        'ChartQA',
        'DocVQA',
        'AI2D',
    ],
    'model_specs': ['price', 'context_window', 'param_count', 'architecture'],
    'community_score': ['lm_sys_arena_score', 'hf_score']
} 
