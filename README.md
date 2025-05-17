<div align="center"> 

<img src="https://github.com/LSeu-Open/AIEnhancedWork/blob/main/Images/LLMScoreEngine.png">

<br>
<br>

***A comprehensive system for evaluating and scoring large language models based on multiple criteria.***

<br>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat)](https://github.com/LSeu-Open/LLMScoreEngine/blob/main/LICENSE)
![LastCommit](https://img.shields.io/github/last-commit/LSeu-Open/LLMScoreEngine?style=flat)
![LastRelease](https://img.shields.io/github/v/release/LSeu-Open/LLMScoreEngine?style=flat)

***Version v0.4.0-beta is currently in development, featuring a broader and significantly more complex approach.***

</div>

<br>

## Overview

This project provides tools for scoring and comparing large language models based on the following criteria:

Originally developed within the [AIenhancedWork](https://github.com/LSeu-Open/AIEnhancedWork) repository to evaluate models in the LLMs section, it has now been migrated to this dedicated project for improved organization, scalability, and focus.

- **Entity benchmarks** (30 points max)
- **Dev benchmarks** (30 points max)
- **Community score** (20 points max)
- **Technical specifications** (20 points max)

The final score is calculated out of 100 points (if you want to have a detailed breakdown of the scoring framework, please refer to the [scoring_framework_development_notes.md](https://github.com/LSeu-Open/AIEnhancedWork/blob/main/Scoring/dev_ideas/scoring_framework_development_notes.md) file).

Please note that this is a beta version and the scoring system is subject to change.

To help us refine and improve LLMScoreEngine during this beta phase, we actively encourage user feedback, bug reports, and contributions to help us refine and improve LLMScoreEngine. Please feel free to [open an issue](https://github.com/LSeu-Open/LLMScoreEngine/issues) or [contribute](CONTRIBUTING.md) to the project. Make sure to respect the [Code of Conduct](CODE_OF_CONDUCT.md).

## Project Structure

```text
LLMScoreEngine/
├── config/                    # Configuration files
│   └── scoring_config.py      # Scoring parameters and thresholds
├── model_scoring/             # Main package
│   ├── core/                  # Core functionality
│   ├── data/                  # Data handling
│   ├── scoring/               # Scoring logic
│   ├── utils/                 # Utility functions
│   └── run_scoring.py         # Script for running scoring programmatically
├── Models/                    # Model data directory (Create this manually)
├── Results/                   # Results directory (Created automatically)
├── tests/                     # Unit and integration tests
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

- Inside the `Models` directory, create a JSON file for each model you want to score (e.g., `Deepseek-R1.json`).
- The filename (without the `.json` extension) should precisely match the model identifier you plan to use.
- Avoid any blank spaces in the model name if you want to score it using the command line.
- Populate each JSON file according to the [Models Data Format](#models-data-format).

### Models Data Format

Models data should be stored as JSON files in the `Models` directory, with the following structure:

```json
{
    "entity_benchmarks": {
        "artificial_analysis": null,
        "OpenCompass": null,
        "LLM Explorer": null,
        "Livebench": null,
        "open_llm": null,
        "UGI Leaderboard": null,
        "big_code_bench": null,
        "EvalPlus Leaderboard": null,
        "Dubesord_LLM": null,
        "Open VLM": null
    },
    "dev_benchmarks": {
        "MMLU": null, 
        "MMLU Pro": null, 
        "BigBenchHard": null,
        "GPQA diamond": null, 
        "DROP": null, 
        "HellaSwag": null,
        "Humanity's Last Exam": null,
        "ARC-C": null,
        "Wild Bench": null,
        "MT-bench": null,
        "IFEval": null,
        "Arena-Hard": null,
        "MATH": null,
        "GSM-8K": null,
        "AIME": null,
        "HumanEval": null,
        "MBPP": null,
        "LiveCodeBench": null,
        "Aider Polyglot": null,
        "SWE-Bench": null,
        "SciCode": null,
        "MGSM": null,
        "MMMLU": null,
        "C-Eval or CMMLU": null,
        "AraMMLu": null,
        "LongBench": null,
        "RULER 128K": null,
        "RULER 32K": null,
        "MTOB": null,
        "BFCL": null,
        "AgentBench": null,
        "Gorilla Benchmark": null,
        "ToolBench": null,
        "MINT": null,
        "MMMU": null,
        "Mathvista": null,
        "ChartQA": null,
        "DocVQA": null,
        "AI2D": null
    },
    "community_score": {
        "lm_sys_arena_score": null,
        "hf_score": null
    },
    "model_specs": {
        "price": null,
        "context_window": null,
        "param_count": null,
        "architecture": null
    }
}
```

Fill the null values with the actual data. While you don't need to fill all values, the following fields are mandatory:

- `model_specs` (all subfields: price, context_window, param_count, architecture)
- `community_score` (at least one subfield: lm_sys_arena_score, hf_score)
- At least one benchmark score in `entity_benchmarks`
- At least one benchmark score in `dev_benchmarks`

All other fields are optional and can remain null if data is not available.

#### Where to find the data ?

- `entity_benchmarks` :

* [Artificial Analysis](https://artificialanalysis.ai/)
* [OpenCompass](https://rank.opencompass.org.cn/home)
* [LLM Explorer](https://llm.extractum.io/list/)
* [Livebench](https://livebench.ai/#/)
* [Open LLM](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/)
* [UGI Leaderboard](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)
* [Big Code Bench](https://bigcode-bench.github.io/)
* [EvalPlus Leaderboard](https://evalplus.github.io/leaderboard.html)
* [Dubesord_LLM](https://dubesor.de/benchtable)
* [Open VLM](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)

- `dev_benchmarks` : You usually find the data on the model's page on the provider's website or on the model's page on the [Hugging Face](https://huggingface.co/) website.

- `community_score` : You will find Elo on the [LM-SYS Arena Leaderboard](https://beta.lmarena.ai/leaderboard) and use the 'hf_score.py' script to get huggingface score.

You can find the 'hf_score.py' script in the **model_scoring/scoring** folder.

**Note**: To use the 'hf_score.py' script, you will need to install the 'huggingface_hub' library if it's not already installed.

```bash
pip install huggingface_hub
```

then you can use the script to get the huggingface score.

Make sure to use the correct model name as it is written on the [Model's page on the Hugging Face](https://huggingface.co/) website. 

For example, the model name for the 'DeepSeek-R1' model is 'deepseek-ai/DeepSeek-R1'.

```bash
python hf_score.py deepseek-ai/DeepSeek-R1
```

- `model_specs` : You will find the price on the model's page on the provider's website or on the model's page on the [Hugging Face](https://huggingface.co/) website. Some of this data can also be found on the [Artificial Analysis](https://artificialanalysis.ai/) website.

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

Results will be stored as JSON files in the `Results` directory, with the following structure (example for Deepseek-R1):

```json
{
    "model_name": "Deepseek-R1",
    "scores": {
        "entity_score": 18.84257142857143,
        "dev_score": 23.063999999999997,
        "external_score": 41.906571428571425,
        "community_score": 16.76,
        "technical_score": 16.95878387917363,
        "final_score": 75.63,
        "avg_performance": 73.21368421052631
    },
    "input_data": {
        "entity_benchmarks": {
            "artificial_analysis": 0.6022,
            "OpenCompass": 0.867,
            "LLM Explorer": 0.59,
            "Livebench": 0.7249,
            "open_llm": null,
            "UGI Leaderboard": 0.5565,
            "big_code_bench": 0.35100000000000003,
            "EvalPlus Leaderboard": null,
            "Dubesord_LLM": 0.705,
            "Open VLM": null
        },
        "dev_benchmarks": {
            "MMLU": 0.9079999999999999,
            "MMLU Pro": 0.84,
            "BigBenchHard": null,
            "GPQA diamond": 0.715,
            "DROP": 0.922,
            "HellaSwag": null,
            "Humanity's Last Exam": null,
            "ARC-C": null,
            "Wild Bench": null,
            "MT-bench": null,
            "IFEval": 0.833,
            "Arena-Hard": 0.9229999999999999,
            "MATH": 0.973,
            "GSM-8K": null,
            "AIME": 0.7979999999999999,
            "HumanEval": null,
            "MBPP": null,
            "LiveCodeBench": 0.659,
            "Aider Polyglot": 0.5329999999999999,
            "SWE-Bench": 0.49200000000000005,
            "SciCode": null,
            "MGSM": null,
            "MMMLU": null,
            "C-Eval or CMMLU": 0.9179999999999999,
            "AraMMLu": null,
            "LongBench": null,
            "RULER 128K": null,
            "RULER 32K": null,
            "MTOB": null,
            "BFCL": null,
            "AgentBench": null,
            "Gorilla Benchmark": null,
            "ToolBench": null,
            "MINT": null,
            "MMMU": null,
            "Mathvista": null,
            "ChartQA": null,
            "DocVQA": null,
            "AI2D": null
        },
        "community_score": {
            "lm_sys_arena_score": 1363,
            "hf_score": 9.5
        },
        "model_specs": {
            "price": 0.55,
            "context_window": 128000,
            "param_count": 685,
            "architecture": "moe"
        }
    }
}
```

## License

This project is licensed under the MIT License - see the [LICENSE-CODE.md](https://github.com/LSeu-Open/AIEnhancedWork/blob/main/LICENSE-CODE.md) file for details.

<br>

<div align="center">

[⬆️ Back to Top](#overview)

</div>

<br>
