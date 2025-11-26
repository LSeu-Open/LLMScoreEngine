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

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [New features in Beta v0.7](#new-features-in-beta-v07)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Model Data Setup](#model-data-setup)
  - [Models Data Format](#models-data-format)
    - [Where to find the data ?](#where-to-find-the-data-)
- [Usage](#usage)
  - [Command-Line Usage (Recommended)](#command-line-usage-recommended)
    - [🚀 Fill Benchmark Pipeline (New)](#-fill-benchmark-pipeline-new)
    - [Visualize All Your Models Scores in One Powerful Report](#visualize-all-your-models-scores-in-one-powerful-report)
    - [Command-Line Options](#command-line-options)
  - [Programmatic Usage](#programmatic-usage)
  - [Interactive Assistant Shell (In Development)](#interactive-assistant-shell-in-development)
    - [Key UX Enhancements](#key-ux-enhancements)
    - [Accessibility \& Inclusivity](#accessibility--inclusivity)
    - [Keyboard Shortcuts](#keyboard-shortcuts)
- [Results Data Format](#results-data-format)
- [Testing Strategy \& Quality Assurance](#testing-strategy--quality-assurance)
  - [Test Layers](#test-layers)
  - [CI Pipeline](#ci-pipeline)
  - [Running Tests Locally](#running-tests-locally)
- [License](#license)

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

## New features in Beta v0.7

- **Automated Benchmark Pipeline**: A powerful new tool (`fill-benchmark-pipeline`) to automatically populate model benchmarks from external APIs.
- **Improved Reporting**: Updated HTML graph generation with enhanced visualization capabilities, better data handling, and improved interactivity.
- **Scoring Engine Update**: Enhanced technical score calculation to include model input/output pricing.
- **Added command-line option** : Enhanced flexibility with new CLI argument for custom configurations, allowing more granular control.
- **Quality Assurance**: Introduced a comprehensive testing strategy and CI pipeline to enforce performance budgets and prevent regressions.
- **Optimized Architecture**: Streamlined dependencies and updated documentation for a better developer experience.
- **Interactive Shell Overhaul (Preview)**: A completely redesigned `llmscore shell` is in active development, featuring a multi-pane layout and smart dock.

## Project Structure

```text
LLMScoreEngine/
├── config/                    # Configuration files
│   └── scoring_config.py      # Scoring parameters and thresholds
├── model_scoring/             # Main package
│   ├── core/                  # Core functionality (exceptions, types, constants)
│   ├── data/                  # Data handling (loaders, validators)
│   │   ├── loaders.py
│   │   └── validators.py
│   ├── scoring/               # Scoring logic
│   │   ├── hf_score.py
│   │   └── models_scoring.py
│   ├── utils/                 # Utility functions
│   │   ├── config_loader.py
│   │   ├── csv_reporter.py
│   │   └── graph_reporter.py
│   │   └── logging.py
│   ├── __init__.py
│   └── run_scoring.py         # Script for running scoring programmatically
├── tools/                     # Additional tools and utilities
│   └── fill-benchmark-pipeline/  # Automated pipeline to fill model benchmark JSONs
│       ├── llm_benchmark_pipeline.py
│       ├── README.md
│       ├── config_example.yaml
│       └── requirements.txt
├── Models/                    # Model data directory (Create this manually)
├── Results/                   # Results directory (Created automatically)
├── tests/                     # Unit and integration tests
│   ├── config/
│   │   └── test_scoring_config.py
│   ├── data/
│   │   └── test_validators.py
│   ├── scoring/
│   │   ├── test_hf_score.py
│   │   └── test_models_scoring.py
│   ├── utils/
│   │   ├── test_config_loader.py
│   │   └── test_csv_reporter.py
│   │   └── test_graph_reporter.py
│   ├── __init__.py
│   └── test_run_scoring.py
├── LICENSE                    # Project license file
├── README.md                  # This file
├── pyproject.toml             # Project configuration (for build system, linters, etc.)
├── requirements.txt           # Project dependencies
└── score_models.py            # Main command-line scoring script
```

## Installation

**Prerequisites:**

- Python >=3.11 installed
- [uv](https://github.com/astral-sh/uv) installed (recommended for dependency management)

**Step 1:** Clone the repository:

```bash
git clone https://github.com/LSeu-Open/LLMScoreEngine.git
```

**Step 2 (recommended):** Run the automated setup script

- **Unix/macOS:**
  ```bash
  cd LLMScoreEngine
  chmod +x setup.sh   # first time only
  ./setup.sh
  ```

- **Windows (PowerShell or Command Prompt):**
  ```bat
  cd LLMScoreEngine
  setup_windows.bat
  ```

These scripts will verify Python/uv, create `.venv`, install `requirements.txt`, and create the `Models/` and `filled_models/` folders. After they finish, activate the environment with:

- Unix/macOS: `source .venv/bin/activate`
- Windows: `call .venv\Scripts\activate.bat`

**Manual setup (alternative):**

1. Create and activate a virtual environment using uv:
   ```bash
   uv venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix or macOS:
   source .venv/bin/activate
   ```

2. Install dependencies:
   - Standard usage:
     ```bash
     uv pip install -e .
     ```
   - Development/testing:
     ```bash
     uv pip install -e ".[dev]"
     ```
   - Or with pip:
     ```bash
     pip install -r requirements.txt
     pip install -e ".[dev]"
     ```

## Model Data Setup

**Step 1:** Ensure the `Models` directory exists.

- If you used `setup.sh` or `setup_windows.bat`, this directory is already created for you (along with `filled_models`).
- Otherwise, create it manually:
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
        "input_price": null,
        "output_price": null,
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

The recommended workflow is to run the **Fill Benchmark Pipeline** (`tools/fill-benchmark-pipeline/llm_benchmark_pipeline.py launch`). It queries the APIs listed below, hydrates `entity_benchmarks`, `dev_benchmarks`, `community_score`, and `model_specs`, and flags any remaining gaps for manual follow-up. Use the manual sources underneath only when the pipeline cannot retrieve a particular metric or when you are working completely offline.

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

- `dev_benchmarks` : The pipeline pulls from provider metadata and Hugging Face, but you can also read the model's provider or [Hugging Face](https://huggingface.co/) page directly for any scores still missing.

- `community_score` : LMSYS ELO is sourced from the [LM-SYS Arena Leaderboard](https://beta.lmarena.ai/leaderboard). The Fill Benchmark Pipeline automatically fetches the Hugging Face community score plus telemetry whenever `hf_id` is provided, so manual runs of `hf_score.py` are only necessary for offline workflows or custom experiments.

If you need to call the script manually, it lives in **model_scoring/scoring** and requires the `huggingface_hub` dependency:

```bash
pip install huggingface_hub

python model_scoring/scoring/hf_score.py deepseek-ai/DeepSeek-R1
```

- `model_specs` : The pipeline will attempt to infer pricing, context, parameters, and architecture via provider APIs; otherwise, collect the details from the model provider or [Hugging Face](https://huggingface.co/) pages (Artificial Analysis is another good source).

## Usage

### Command-Line Usage (Recommended)

You can run the scoring script from your terminal.

**Score specific models:**

Provide the names of the models (without the `.json` extension) as arguments:

```bash
python score_models.py ModelName1 ModelName2
```

**Score all models:**

Use the `--all` flag to score all models present in the `Models` directory.

```bash
python score_models.py --all
```

To read models from a different folder, pass `--models-dir` alongside `--all` (or any explicit model list):

```bash
python score_models.py --all --models-dir CustomModels/
```

#### 🚀 Fill Benchmark Pipeline (New)

The `tools/fill-benchmark-pipeline/` directory contains a powerful new automated pipeline for filling model benchmark JSON files with data from multiple API sources. This is the recommended tool for preparing model data.

**Features:**
- 🚀 **Interactive CLI** with guided prompts and automatic model detection
- 🔧 **Multi-API Integration** (Artificial Analysis, Hugging Face)
- ✅ **Input Validation** using Pydantic models
- ⚡ **Rate Limiting & Retry Logic** with exponential backoff
- 📊 **Rich Progress Reporting** and coverage statistics

**Installation:**
```bash
pip install -e .[fill-benchmark-pipeline]
```

**Usage:**
```bash
# Interactive mode (recommended)
python tools/fill-benchmark-pipeline/llm_benchmark_pipeline.py launch

# Process with config file
python tools/fill-benchmark-pipeline/llm_benchmark_pipeline.py --config config.yaml
```

For detailed usage instructions, see the [pipeline README](docs/FILL_BENCH_README.md).

#### Visualize All Your Models Scores in One Powerful Report

Discover everything you need to evaluate performance and efficiency at a glance:

- **Interactive Leaderboard**: Rank all your models with smart filters for quick comparisons.
- **Insightful Visualizations**: Explore key metrics including:
  - Performance vs. Parameter Count  
  - Score Composition  
  - Cost Analysis  
  - Architecture Distribution
- **Cost-Efficiency Leaderboard**: Identify the best-performing models relative to their cost.
- **Model Comparison Tool**: Easily compare multiple models side by side.

All insights in one unified, actionable report — no more scattered data.

Create this comprehensive report from your models in just two commands:

1. Run a silent, CSV-exported model evaluation :

```bash
python score_models.py --all --quiet --csv
```

2. Generate visualizations and the final report :

```bash
python score_models.py --graph
```
<br>

#### Command-Line Options

You can customize the scoring process with the following optional flags:

| Flag                 | Description                                                                                             | Example                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| `--all`              | Score all models found in the `Models/` directory (or the folder set via `--models-dir`).               | `python score_models.py --all`                      |
| `--models-dir PATH`  | Directory containing model JSON files (defaults to `./Models`).                                         | `python score_models.py --all --models-dir CustomModels/` |
| `--quiet`            | Suppress all informational output and only print the final scores in the console. Useful for scripting. | `python score_models.py --all --quiet`              |
| `--config <path>`    | Path to a custom Python configuration file to override the default scoring parameters.                  | `python score_models.py ModelName --config my_config.py` |
| `--csv`              | Generate a CSV report from existing results.                                                            | `python score_models.py --csv`                      |
| `--graph`            | Generate a graph report from existing csv report.                                                       | `python score_models.py --graph`                    |
| `--skip-hf-score`    | (Fill pipeline) Skip Hugging Face telemetry calls if you need to conserve quota or run offline.         | `python tools/fill-benchmark-pipeline/llm_benchmark_pipeline.py --skip-hf-score ...` |

### Programmatic Usage

You can also call the scoring functions directly from your Python code. Import the necessary functions from `model_scoring.scoring.score_models` and use them programmatically.

### Interactive Assistant Shell (In Development)

> **⚠️ Note:** The Interactive Shell is currently in **active development** and is considered an experimental feature. APIs and UI elements may change frequently.

The **LLMScore Shell** (`llmscore shell`) has undergone a comprehensive UI/UX overhaul to support daily evaluation workflows with a modern, multi-pane console experience. This unified interface brings together execution, monitoring, and context management into a single ergonomic workspace.

#### Key UX Enhancements

- **Multi-Pane Workspace**: A responsive 3-column layout featuring:
  - **Command Stream**: The central hub for execution and logs.
  - **Timeline Pane**: A chronological feed of events, actions, and system state.
  - **Context Pane**: Persistent view of active configurations, pinned metrics, and scratchpad notes.
- **Smart Dock**: A fixed, always-visible input area that anchors the bottom of the screen, hosting the prompt, autocomplete suggestions, and quick-action status chips.
- **Command Palette 2.0**: A keyboard-centric (`Ctrl+K`) palette for quick navigation, command execution, and workspace management, featuring fuzzy search and grouped actions.
- **Structured Output Cards**: Rich, interactive result cards that replace raw text logs, offering organized summaries, copy-to-clipboard buttons, and export options.

#### Accessibility & Inclusivity

We are committed to a shell that works for everyone. Beta v0.7 introduces:
- **Deterministic Focus Order**: Cycle seamlessly between panes (Timeline → Context → Command → Dock) using `Ctrl+P` with screen-reader announcements.
- **Reduced Motion Mode**: A dedicated configuration flag to disable spinners and sliding animations for a static, distraction-free experience.
- **Color-Blind Support**: High-contrast palette tokens ensuring readability and distinction for status indicators and charts 

#### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+K` | Open Command Palette |
| `Ctrl+P` | Cycle Pane Focus (Timeline → Context → Stream → Dock) |
| `Ctrl+T` | Toggle Timeline Pane |
| `Ctrl+C` | Toggle Context Pane |
| `Ctrl+Shift+L` | Clear Screen |

For a complete guide on commands, layout configuration, and troubleshooting, see the [Shell Documentation](docs/SHELL_README.md).

## Results Data Format

Results will be stored as JSON files in the `Results` directory, with the following structure (example for Deepseek-R1):

```json
{
    "model_name": "Deepseek-R1",
    "scores": {
        "entity_score": 18.84,
        "dev_score": 23.06,
        "community_score": 16.76,
        "technical_score": 16.96,
        "final_score": 75.63,
        "avg_performance": 73.21
    },
    "entity_benchmarks": {
        "artificial_analysis": 60.22,
        "OpenCompass": 86.7,
        "LLM Explorer": 59.0,
        "Livebench": 72.49,
        "open_llm": null,
        "UGI Leaderboard": 55.65,
        "big_code_bench": 35.1,
        "EvalPlus Leaderboard": null,
        "Dubesord_LLM": 70.5,
        "Open VLM": null
    },
    "dev_benchmarks": {
        "MMLU": 90.8,
        "MMLU Pro": 84.0,
        "GPQA diamond": 71.5,
        "DROP": 92.2,
        "IFEval": 83.3,
        "Arena-Hard": 92.3,
        "MATH": 97.3,
        "AIME": 79.8,
        "LiveCodeBench": 65.9,
        "Aider Polyglot": 53.3,
        "SWE-Bench": 49.2,
        "C-Eval or CMMLU": 91.8
    },
    "community_score": {
        "lm_sys_arena_score": 1389,
        "hf_score": 8.79
    },
    "model_specs": {
        "input_price": 0.55,
        "output_price": 2.19,
        "context_window": 128000,
        "param_count": 685,
        "architecture": "moe"
    }
}
```

## Testing Strategy & Quality Assurance

We employ a multi-layer testing strategy to ensure stability, accuracy, and performance across the LLMScoreEngine.

### Test Layers
1. **Unit & Contract**: Validates business logic, schemas, and action handlers in isolation.
2. **Integration**: Verifies interactions with the file system, session store, and external APIs.
3. **End-to-End (CLI)**: Validates full user flows, including `shell`, `run`, and `exec` modes.
4. **UI Snapshots**: Regression testing for the Shell UI using `pytest-regressions` to ensure layout stability and visual consistency.
5. **Performance**: Smoke tests for critical paths (`score.batch`, `results.leaderboard`) and concurrency stress tests for automation.

### CI Pipeline
Our GitHub Actions workflow (`perf-accessibility.yml`) enforces quality gates on every push:
- **Static Analysis**: `ruff`, `mypy`.
- **Test Suite**: Runs the full `pytest` suite, including legacy regressions and new shell UI tests.
- **Performance Gates**: Checks for runtime regressions (>20%) and memory leaks.
- **Accessibility**: Verifies focus order and reduced-motion compliance.

### Running Tests Locally
```bash
# Run fast unit tests
pytest tests/scoring tests/data tests/config

# Run Shell UI regression tests
pytest tests/shell/test_layout.py tests/shell/test_output_cards.py --regression-fail-under=100

# Run full suite
pytest
```

For a deep dive into our testing methodology, please refer to [dev_docs/TESTING.md](dev_docs/TESTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/LSeu-Open/LLMScoreEngine/blob/main/LICENSE) file for details.

<br>

<div align="center">

[⬆️ Back to Top](#overview)

</div>

<br>
