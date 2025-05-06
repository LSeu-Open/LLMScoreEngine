<div align="center"> 

<img src="https://github.com/LSeu-Open/AIEnhancedWork/blob/main/Images/LLMScoreEngine.png">

<br>
<br>

***A comprehensive system for evaluating and scoring large language models based on multiple criteria.***

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://github.com/LSeu-Open/LLMScoreEngine/blob/main/LICENSE)
![LastCommit](https://img.shields.io/github/last-commit/LSeu-Open/LLMScoreEngine?style=flat)
![LastRelease](https://img.shields.io/github/v/release/LSeu-Open/LLMScoreEngine?style=flat)

</div>

<br>

## Introduction

This evaluation system provides an objective way to assess Large Language Models (LLMs) using standardized metrics. 

Originally developed within the [AIenhancedWork](https://github.com/LSeu-Open/AIEnhancedWork) repository to evaluate models in the LLMs section, it has now been migrated to this dedicated project for improved organization, scalability, and focus.
 
This project provides tools for scoring and comparing large language models based on the following criteria:

- **Entity benchmarks** (25 points max)
- **Dev benchmarks** (35 points max)
- **Community score** (20 points max)
- **Technical specifications** (20 points max)

The final score is calculated out of 100 points (if you want to have a detailed breakdown of the scoring framework, please refer to the [scoring_framework.md](https://github.com/LSeu-Open/AIEnhancedWork/blob/main/Scoring/scoring_framework.md) file).

Please note that this is a beta version and the scoring system is subject to change.

## Project Structure

```text
LLMScoreEngine/
├── config/                    # Configuration files
│   ├── __init__.py
│   └── scoring_config.py      # Scoring parameters and thresholds
├── model_scoring/             # Main package
│   ├── core/                  # Core functionality
│   ├── data/                  # Data handling
│   ├── scoring/               # Scoring logic
│   ├── utils/                 # Utility functions
│   ├── __init__.py
│   └── run_scoring.py         # Script for running scoring programmatically
├── Models/                    # Model data directory (Create this manually)
├── Results/                   # Results directory (Created automatically)
├── tests/                     # Unit and integration tests
│   ├── __init__.py
│   ├── test_model_scorer.py
│   ├── test_scoring.py
│   └── test_validators.py
├── LICENSE                    # Project license file
├── README.md                  # This file
├── pyproject.toml             # Project configuration (for build system, linters, etc.)
├── requirements.txt           # Project dependencies
└── score_models.py            # Main command-line scoring script
```

## Installation

**Prerequisites:**

- Python 3.x installed (tested on Python 3.13.1)
- [uv](https://github.com/astral-sh/uv) installed (recommended for dependency management)

**Step 1:** Clone the repository:

```bash
git clone https://github.com/LSeu-Open/LLMScoreEngine.git
cd LLMScoreEngine
```

**Step 2:** Create and activate a virtual environment:

Using uv (recommended):

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate
```

**Step 3:** Install the dependencies:

Using uv (recommended):

```bash
uv pip install -e .
```

Or using pip:

```bash
pip install -r requirements.txt
```

## Model Data Setup

**Step 1:** Create the `Models` directory:

```bash
mkdir Models
```

**Step 2:** Add Model Data:

- Inside the `Models` directory, create a JSON file for each model you want to score (e.g., `Athene-V2-Chat.json`).
- The filename (without the `.json` extension) should precisely match the model identifier you plan to use.
- Avoid any blank spaces in the model name if you want to score it using the command line.
- Populate each JSON file according to the [Models Data Format](#models-data-format).

### Models Data Format

Models data should be stored as JSON files in the `Models` directory, with the following structure:

```json
{
    "entity_benchmarks": {
        "artificial_analysis": null,
        "live_code_bench": null,
        "big_code_models": null,
        "open_llm": null
    },
    "dev_benchmarks": {
        "MMLU": null,
        "MMLU Pro": null,
        "BigBench": null,
        "DROP": null,
        "HellaSwag": null,
        "GPQA": null,
        "ARC-C": null,
        "LiveBench": null,
        "LatestEval": null,
        "AlignBench": null,
        "Wild Bench": null,
        "MT-bench": null,
        "IFEval": null,
        "Arena-Hard": null,
        "TruthfulQA": null,
        "MATH": null,
        "GSM-8K": null,
        "MGSM": null,
        "HumanEval": null,
        "HumanEval Plus": null,
        "MBPP": null,
        "MBPP Plus": null,
        "SWE-bench": null,
        "API-Bank": null,
        "BFCL": null,
        "Gorilla Benchmark": null,
        "Nexus": null
    },
    "community_score": null,
    "model_specs": {
        "price": null,
        "context_window": null,
        "param_count": null
    }
}
```

Fill the null values with the actual data. While you don't need to fill all values, the following fields are mandatory:

- `model_specs` (all subfields: price, context_window, param_count)
- `community_score` (must be a numerical value)
- At least one benchmark score in `entity_benchmarks`
- At least one benchmark score in `dev_benchmarks`

All other fields are optional and can remain null if data is not available.

## Usage

### IDE Usage

- Open the `score_models.py` file
- Update the `models` list with the models you want to score (You can add as many models as you want)
- You can run the script in several ways:
  - Click the "Run" button in the top-right corner
  - Use the Run and Debug view (Ctrl+Shift+D or Cmd+Shift+D)
  - Right-click in the editor and select "Run Python File in Terminal"

### Command Line Interface

You can add as many models as you want. Make sure to add the models in the `Models` folder first.

The scoring system can be used via the command line with the following options:

```bash
# Score specific models
python score_models.py model1 model2 model3

# Show help
python score_models.py --help

# Show version
python score_models.py --version

# Enable verbose logging
python score_models.py -v model1 model2
```

### Programmatic Usage

You can also use the scoring system programmatically:

```python
from model_scoring.run_scoring import batch_process_models

# Score specific models
models = ["model1", "model2", "model3"]
batch_process_models(models)

# Or use the main function
from score_models import main
main(models)
```

## Results Data Format

Results will be stored as JSON files in the `Results` directory, with the following structure (example for Athene-V2-Chat):

```json
{
    "model_name": "Athene-V2-Chat",
    "scores": {
        "entity_score": 10.885,
        "dev_score": 25.22264705882353,
        "external_score": 36.10764705882353,
        "community_score": 13.681592039800996,
        "technical_score": 14,
        "final_score": 63.79,
        "avg_performance": 62.097500000000004,
        "size_perf_ratio": 3.0
    },
    "input_data": {
        "entity_benchmarks": {
            "artificial_analysis": 0.401,
            "live_code_bench": null,
            "big_code_models": null,
            "open_llm": 0.4698
        },
        "dev_benchmarks": {
            "MMLU": null,
            "MMLU Pro": 0.737,
            "BigBench": 0.314,
            "DROP": null,
            "HellaSwag": null,
            "GPQA": 0.535,
            "ARC-C": null,
            "LiveBench": null,
            "LatestEval": null,
            "AlignBench": null,
            "Wild Bench": null,
            "MT-bench": null,
            "IFEval": 0.8320000000000001,
            "Arena-Hard": 0.8490000000000001,
            "TruthfulQA": null,
            "MATH": 0.83,
            "GSM-8K": null,
            "MGSM": null,
            "HumanEval": null,
            "HumanEval Plus": null,
            "MBPP": null,
            "MBPP Plus": null,
            "SWE-bench": null,
            "API-Bank": null,
            "BFCL": null,
            "Gorilla Benchmark": null,
            "Nexus": null
        },
        "community_score": 1275,
        "model_specs": {
            "price": 1,
            "context_window": 128000,
            "param_count": 72.7
        }
    }
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/LSeu-Open/LLMScoreEngine/blob/main/LICENSE) file for details.

<br>
<br>
