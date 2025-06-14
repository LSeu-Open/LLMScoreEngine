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
Static Application Constants.

This module defines static constants for the model scoring application.
These values are not expected to change frequently and are related to the
application's structure and data validation rules, not the scoring algorithm itself.

This includes directory paths for models and results, and the validation schemas
(REQUIRED_SECTIONS) that define the expected structure of input model data.
For tunable scoring parameters (weights, thresholds), see `config.scoring_config`.
"""

# ------------------------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------------------------

# Directory and file constants
MODELS_DIR = "Models"
RESULTS_DIR = "Results"
LOG_FILE = "model_scoring.log"

# Required sections and fields that must be present in model JSON files for validation
REQUIRED_SECTIONS = {
    'entity_benchmarks': [
        'artificial_analysis',
        'OpenCompass',
        'Dubesord_LLM',
        'LLM Explorer',
        'Livebench',
        'open_llm',
        'UGI Leaderboard',
        'big_code_bench',
        'EvalPlus Leaderboard',        
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
