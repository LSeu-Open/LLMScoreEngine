# LLM Benchmark Data Pipeline

Automated Python pipeline to fill model benchmark JSONs using multiple API sources, designed for batch processing of LLM model evaluations.

## Features

✅ **Multiple Usage Modes**
- 🚀 **Interactive CLI**: Guided step-by-step processing with automatic model detection
- 🔧 **Advanced CLI**: Full command-line control with config files and overrides
- 🐍 **Programmatic API**: Python library integration with validation and error handling

✅ **Multi-Source API Integration**
- Artificial Analysis API (primary source for benchmarks, pricing, specs)
- Hugging Face API (model cards, parameters, leaderboard data)
- Extensible architecture for adding more sources

✅ **Production-Ready**
- Automatic rate limiting and retry logic with exponential backoff
- Comprehensive error handling with custom exceptions
- Input validation using Pydantic models
- Batch processing support with continuation on errors
- Coverage reporting for data completeness
- Structured logging with configurable levels

✅ **User-Friendly**
- Interactive mode with emojis and clear prompts
- Automatic JSON file detection and model name extraction
- Rich progress reporting and final summaries
- Safe file path handling and validation
- Configuration files (YAML/JSON) for complex setups

✅ **Scalable**
- Process dozens or hundreds of models efficiently
- Parallel-ready architecture (can be extended)
- Minimal API calls with intelligent caching
- Configurable rate limits and timeouts

## Installation

### Option 1: Install from LLMScoreEngine (Recommended)

If you're working within the LLMScoreEngine project:

```bash
# Install the main project with fill-benchmark-pipeline dependencies
pip install -e .[fill-benchmark-pipeline]

# Or install all dependencies from requirements.txt
pip install -r requirements.txt
```

### Option 2: Standalone Installation

If using this pipeline independently:

```bash
# Clone or download the pipeline files
# Install dependencies
pip install -r requirements.txt
```

### 3. Set up API keys

Copy `.env.example` to `.env` and add your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your keys:

```
ARTIFICIAL_ANALYSIS_API_KEY=your_actual_key
HUGGINGFACE_API_KEY=your_hf_token  # optional
```

#### Getting API Keys

- **Artificial Analysis**: Register at https://artificialanalysis.ai/api
  - Free tier: 1000 requests/day
  - Provides: benchmarks, pricing, context windows, architectures

- **Hugging Face**: Get token from https://huggingface.co/settings/tokens
  - Optional, but increases rate limits
  - Provides: model cards, parameter counts, some leaderboard data

## Configuration

### Configuration Files (Optional)

For complex setups or automation, create a configuration file (`pipeline_config.yaml` or `pipeline_config.json`):

```yaml
# Example: pipeline_config.yaml
template_path: "GLM-4.6.json"
output_dir: "./filled_models"

# API Configuration
artificial_analysis_key: null  # Uses env var ARTIFICIAL_ANALYSIS_API_KEY
huggingface_key: null          # Uses env var HUGGINGFACE_API_KEY

# Rate limiting (requests per second)
rate_limit_aa: 1.0
rate_limit_hf: 0.5

# Retry and timeout configuration
max_retries: 3
retry_backoff_factor: 2.0
timeout: 30

# Processing options
continue_on_error: true

# Models to process (for programmatic usage)
models:
  - name: "GLM-4.6"
    hf_id: "zai-org/GLM-4.6"
  - name: "Qwen2.5-72B-Instruct"
    hf_id: "Qwen/Qwen2.5-72B-Instruct"
```

Then run: `python llm_benchmark_pipeline.py --config pipeline_config.yaml`

## Usage

### 🚀 Interactive CLI Mode (Recommended)

The easiest way to use the pipeline is through the interactive CLI mode:

```bash
# Launch interactive pipeline
python llm_benchmark_pipeline.py launch
```

**Interactive Workflow:**
1. **API Keys**: Enter your Artificial Analysis and Hugging Face API keys (or press Enter to skip)
2. **Input Folder**: Specify folder containing model JSON files (defaults to current directory)
3. **Model Detection**: Pipeline automatically scans for JSON files and detects model names
4. **Confirmation**: Review detected models and select which ones to process
5. **Output Folder**: Choose where to save results (defaults to `./filled_models`)
6. **Processing**: Choose verbose mode for detailed progress updates
7. **Results**: Rich recap showing success/failure counts and saved files

**Benefits:**
- ✅ No configuration files needed
- ✅ Guided step-by-step process
- ✅ Automatic model detection from JSON files
- ✅ Visual progress indicators
- ✅ Comprehensive result summary
- ✅ Error recovery and continuation

**Example Output:**
```
🤖 LLM Benchmark Pipeline - Interactive Mode
============================================================
🔑 API Keys
Artificial Analysis API key (or ENTER to skip):
Hugging Face API key (or ENTER to skip):

📁 Input Folder
Input folder containing model JSONs (default: ./): ./Models

🔍 Found 162 JSON file(s)

📋 Detected models:
  1. GLM-4.6.json -> GLM-4.6
  2. Qwen2.5-72B.json -> Qwen2.5-72B
  3. DeepSeek-V2.5.json -> DeepSeek-V2.5
  ...

✅ Confirmation
Process all detected files? [Y/n]: y

📂 Output Folder
Output folder for results (default: ./filled_models): ./filled_models

Enable verbose updates during processing? [y/N]: y

🚀 Starting processing of 162 file(s)...

[1/162] Processing: GLM-4.6.json
   📝 Processing model: GLM-4.6
   ✅ Saved to: ./filled_models/GLM-4.6_filled.json
...
```

### Advanced CLI Usage

For automated workflows or CI/CD pipelines, use the advanced CLI options:

```bash
# Process with config file
python llm_benchmark_pipeline.py --config pipeline_config.yaml

# Override specific settings
python llm_benchmark_pipeline.py --template GLM-4.6.json --models models.json --output-dir ./results

# Full command with all options
python llm_benchmark_pipeline.py \
  --template model_template.json \
  --models '[{"name": "GLM-4.6", "hf_id": "zai-org/GLM-4.6"}]' \
  --output-dir ./output \
  --max-retries 5 \
  --rate-limit-aa 2.0 \
  --rate-limit-hf 1.0 \
  --verbose
```

**CLI Options:**
- `--config, -c`: Load settings from YAML/JSON config file
- `--template, -t`: Path to template JSON file
- `--models, -m`: Path to models JSON file or inline JSON string
- `--output-dir, -o`: Output directory (default: ./filled_models)
- `--max-retries`: Maximum retry attempts (default: 3)
- `--rate-limit-aa`: AA API rate limit (req/s, default: 1.0)
- `--rate-limit-hf`: HF API rate limit (req/s, default: 0.5)
- `--timeout`: Request timeout in seconds (default: 30)
- `--aa-key`: Artificial Analysis API key
- `--hf-key`: Hugging Face API key
- `--log-level`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--verbose, -v`: Enable verbose logging

### Programmatic Usage

For integration into other Python projects:

#### Single Model Processing

```python
from llm_benchmark_pipeline import LLMBenchmarkPipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    template_path='GLM-4.6.json',
    output_dir='./results',
    artificial_analysis_key="your_aa_key",
    huggingface_key="your_hf_key"  # optional
)

# Initialize pipeline
pipeline = LLMBenchmarkPipeline(config)

# Process single model
template = pipeline.load_template(config.template_path)
filled_data = pipeline.fill_model_data(template, {'name': 'GLM-4.6', 'hf_id': 'zai-org/GLM-4.6'})
pipeline.save_result(filled_data, './results/GLM-4.6_filled.json')
```

#### Batch Processing

```python
from llm_benchmark_pipeline import LLMBenchmarkPipeline, PipelineConfig, ModelInfo
import os

# Create configuration
config = PipelineConfig(
    template_path='GLM-4.6.json',
    output_dir='./batch_results',
    artificial_analysis_key=os.getenv('ARTIFICIAL_ANALYSIS_API_KEY'),
    huggingface_key=os.getenv('HUGGINGFACE_API_KEY')
)

# Initialize pipeline
pipeline = LLMBenchmarkPipeline(config)

# Define models using validated ModelInfo objects
models = [
    ModelInfo(name='GLM-4.6', hf_id='zai-org/GLM-4.6'),
    ModelInfo(name='Qwen2.5-72B', hf_id='Qwen/Qwen2.5-72B'),
    ModelInfo(name='DeepSeek-V2.5', hf_id='deepseek-ai/DeepSeek-V2.5'),
    # Or use dictionaries (auto-validated)
    {'name': 'Llama-3.1-405B', 'hf_id': 'meta-llama/Llama-3.1-405B'},
]

# Process batch
results = pipeline.process_batch(models)

# Check results
for result in results:
    print(f"{result['model']}: {result['status']}")
```

## API Coverage

### What the Pipeline Can Fill via APIs

| Data Category | Coverage | Source |
|--------------|----------|--------|
| **Model Specs** | 80-90% | Artificial Analysis, HF |
| Context window | ✓ | Artificial Analysis |
| Parameter count | ✓ | Both sources |
| Architecture | ✓ | Artificial Analysis |
| Pricing | ✓ | Artificial Analysis |
| **Major Benchmarks** | 50-70% | Artificial Analysis |
| MMLU Pro | ✓ | Artificial Analysis |
| GPQA Diamond | ✓ | Artificial Analysis |
| MATH | ✓ | Artificial Analysis |
| AIME | ✓ | Artificial Analysis |
| HumanEval | ✓ | Artificial Analysis |
| MBPP | ✓ | Artificial Analysis |
| LiveCodeBench | ✓ | Artificial Analysis |
| SciCode | ✓ | Artificial Analysis |
| **Community Scores** | 30-50% | Limited |
| LMSys Arena ELO | Partial | Manual/scraping needed |
| HuggingFace Score | ✓ | Hugging Face API |

### What Still Requires Manual Input or Scraping

- Niche/emerging benchmarks (AgentBench, ToolBench, MINT)
- Some entity leaderboards (OpenCompass, UGI, Dubesord)
- Multimodal benchmarks (MMMU, MathVista, ChartQA) for text-only models
- Very recent benchmark results not yet in APIs
- Some proprietary benchmark results

## Output Format

The pipeline fills your JSON template with this structure:

```json
{
  "entity_benchmarks": {
    "artificial_analysis": 44.7,
    "Livebench": 71.22,
    ...
  },
  "dev_benchmarks": {
    "MMLU Pro": 78.4,
    "GPQA diamond": 81.0,
    ...
  },
  "community_score": {
    "lm_sys_arena_score": 1398,
    "hf_score": null
  },
  "model_specs": {
    "input_price": 0.6,
    "output_price": 2.2,
    "context_window": 200000,
    "param_count": "355B total / 32B active",
    "architecture": "Mixture-of-Experts (MoE)"
  }
}
```

## Extending the Pipeline

### Adding a New API Source

```python
class NewAPIClient(APIClient):
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(
            base_url="https://api.newservice.com",
            api_key=api_key,
            rate_limit=1.0
        )

    def get_benchmarks(self, model_name: str) -> Dict:
        data = self.get(f"/models/{model_name}/benchmarks")
        # Parse and return benchmark data
        return data
```

Then integrate into the pipeline's `fill_model_data` method.

### Adding Custom Benchmark Mappings

Edit the `benchmark_mapping` dict in `ArtificialAnalysisClient.extract_benchmarks()`:

```python
benchmark_mapping = {
    'mmlu_pro': 'MMLU Pro',
    'your_new_benchmark': 'Your Benchmark Name',
    # Add more mappings
}
```

## Performance & Rate Limits

- **Artificial Analysis**: 1 request/second (free tier), 1000/day
- **Hugging Face**: ~2 requests/second (without token), higher with token
- **Expected throughput**: 50-100 models per hour (with delays)
- **Batch of 100 models**: ~30-45 minutes

## Troubleshooting

### Interactive Mode Issues

#### "No JSON files found in folder"
- Verify the input folder path exists and contains `.json` files
- Check file permissions and ensure files are readable
- Use absolute paths if relative paths don't work

#### "Could not read file" errors
- Ensure JSON files are valid (use a JSON validator)
- Check file encoding (should be UTF-8)
- Verify file permissions

#### Model name detection issues
- Pipeline tries to extract names from `model_specs.model_name` or filename
- For custom JSON structures, ensure model name is in a `name` field
- Check logs for detection failures

### API and Data Issues

#### "API request failed" errors
- Check your API keys in `.env` or when prompted in interactive mode
- Verify rate limits haven't been exceeded (wait and retry)
- Check internet connection and firewall settings

#### "Authentication failed" errors
- Verify API keys are correct and not expired
- Check if you've exceeded API quotas
- For Hugging Face, ensure token has appropriate permissions

#### Low coverage percentage
- Model might be too new/obscure for API databases
- Try alternative model names (check AA website for exact naming)
- Some benchmarks require manual entry (see API Coverage section)

#### "Model not found" warnings
- Verify exact model name (case-sensitive, including spaces/dashes)
- Check if model exists on Artificial Analysis website
- Try providing alternative names in batch config
- For interactive mode: verify the detected names are correct

### Configuration Issues

#### "Configuration validation failed"
- Check YAML/JSON syntax in config files
- Verify all required fields are present
- Check data types (numbers vs strings, etc.)

#### CLI argument errors
- Use `python llm_benchmark_pipeline.py --help` to see available options
- Ensure file paths exist when specified
- Check for typos in option names

### Performance Issues

#### Slow processing
- Reduce rate limits if hitting API throttling
- Increase timeout values for slow connections
- Process smaller batches to identify bottlenecks

#### Memory usage
- Large JSON files are loaded entirely into memory
- Consider splitting very large model collections
- Monitor system resources during processing

## Best Practices

### For Interactive Mode
1. **Start with the interactive mode** (`python llm_benchmark_pipeline.py launch`) for first-time use
2. **Use a small input folder** initially to test API keys and connections
3. **Review detected models** carefully before confirming processing
4. **Choose verbose mode** to see detailed progress during processing
5. **Note the output location** for easy access to results

### For All Usage Modes
1. **Test with a single model** first to verify API keys and connections
2. **Use environment variables** for API keys (never commit `.env` files)
3. **Monitor coverage reports** to identify data gaps and API limitations
4. **Configure appropriate rate limits** based on your API tiers
5. **Enable continue_on_error** for batch processing to handle individual failures gracefully
6. **Use configuration files** for complex or repeated setups
7. **Keep backup copies** of original JSON files before processing

### For Automation
1. **Use config files** for CI/CD pipelines and automated workflows
2. **Set up proper logging** with appropriate log levels for monitoring
3. **Configure timeouts** based on your network conditions
4. **Monitor API usage** to stay within rate limits and quotas
5. **Implement result validation** to ensure data quality in automated runs

## License

This pipeline code is provided as-is for research and data collection purposes.

## Contributing

To add new data sources or improve the pipeline:
1. Extend the `APIClient` base class
2. Add benchmark/spec extraction methods
3. Integrate into `LLMBenchmarkPipeline.fill_model_data()`
4. Update coverage calculations

## Support

For issues with:
- **Pipeline code**: Review error logs, check API keys
- **API access**: Contact the respective API provider
- **Data accuracy**: Verify against source websites
