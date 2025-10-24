#!/usr/bin/env python3
"""
LLM Benchmark Data Pipeline
=============================
Automated pipeline to fill model benchmark JSONs using multiple API sources.

Supports:
- Artificial Analysis API (benchmark scores, pricing, specs)
- Hugging Face API (model cards, parameters, leaderboard data)
- Batch processing of multiple model JSONs

Requirements:
    pip install requests pandas tqdm python-dotenv pydantic pyyaml tenacity

Environment variables (.env file):
    ARTIFICIAL_ANALYSIS_API_KEY=your_key_here
    HUGGINGFACE_API_KEY=your_key_here  # optional, for rate limits
"""

import json
import os
import time
import argparse
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import requests
from dataclasses import dataclass, asdict
import logging
from pydantic import BaseModel, ValidationError, field_validator, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import functools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Pydantic Models for Validation
class ModelBenchmarkTemplate(BaseModel):
    """Pydantic model for validating model benchmark template structure"""
    entity_benchmarks: Dict[str, Optional[Union[float, int]]] = Field(..., description="Entity-provided benchmark scores")
    dev_benchmarks: Dict[str, Optional[Union[float, int]]] = Field(..., description="Development benchmark scores")
    community_score: Dict[str, Optional[Union[float, int]]] = Field(..., description="Community evaluation scores")
    model_specs: Dict[str, Optional[Union[float, int, str]]] = Field(..., description="Model specifications")

    @field_validator('entity_benchmarks', 'dev_benchmarks', 'community_score', 'model_specs')
    @classmethod
    def validate_dict_not_empty(cls, v):
        if not v:
            raise ValueError('Dictionary cannot be empty')
        return v


class ModelInfo(BaseModel):
    """Pydantic model for validating model information"""
    name: str = Field(..., min_length=1, description="Model name")
    hf_id: Optional[str] = Field(None, description="Hugging Face model ID")

    @field_validator('name')
    @classmethod
    def validate_model_name(cls, v):
        if not v.strip():
            raise ValueError('Model name cannot be empty or whitespace')
        # Basic sanitization check
        dangerous_chars = ['/', '\\', '..', '<', '>', '|', '*', '?']
        if any(char in v for char in dangerous_chars):
            raise ValueError(f'Model name contains dangerous characters: {dangerous_chars}')
        return v.strip()

    @field_validator('hf_id')
    @classmethod
    def validate_hf_id(cls, v):
        if v is None:
            return v
        if not v.strip():
            raise ValueError('Hugging Face ID cannot be empty string, use None instead')
        # Basic HF ID format validation (org/model)
        if '/' not in v or v.startswith('/') or v.endswith('/'):
            raise ValueError('Hugging Face ID should be in format: org/model')
        return v.strip()


class PipelineConfig(BaseModel):
    """Configuration model for the pipeline"""
    artificial_analysis_key: Optional[str] = Field(None, description="Artificial Analysis API key")
    huggingface_key: Optional[str] = Field(None, description="Hugging Face API key")
    template_path: str = Field(..., description="Path to template JSON file")
    output_dir: str = Field("./filled_models", description="Output directory for results")
    rate_limit_aa: float = Field(1.0, gt=0, description="Rate limit for AA API (requests per second)")
    rate_limit_hf: float = Field(0.5, gt=0, description="Rate limit for HF API (requests per second)")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts for API calls")
    retry_backoff_factor: float = Field(2.0, gt=0, description="Exponential backoff factor")
    timeout: int = Field(30, gt=0, description="Request timeout in seconds")
    continue_on_error: bool = Field(True, description="Continue processing other models if one fails")

    @field_validator('template_path', 'output_dir')
    @classmethod
    def validate_paths(cls, v):
        path = Path(v)
        if 'template_path' in cls.model_fields and not path.exists():
            raise ValueError(f'Path does not exist: {v}')
        return str(path)


@dataclass
class ModelSpecs:
    """Model specifications data structure"""
    input_price: Optional[float] = None
    output_price: Optional[float] = None
    context_window: Optional[int] = None
    param_count: Optional[str] = None
    architecture: Optional[str] = None


# Custom Exceptions
class APIError(Exception):
    """Base exception for API-related errors"""
    pass


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded"""
    pass


class APIAuthenticationError(APIError):
    """Raised when API authentication fails"""
    pass


class APITimeoutError(APIError):
    """Raised when API request times out"""
    pass


def create_retry_decorator(max_retries: int, backoff_factor: float):
    """Create a retry decorator with configurable parameters"""
    return retry(
        stop=stop_after_attempt(max_retries + 1),  # +1 for initial attempt
        wait=wait_exponential(multiplier=backoff_factor, min=1, max=60),
        retry=retry_if_exception_type((
            requests.exceptions.RequestException,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
            APIError
        )),
        before_sleep=lambda retry_state: logger.warning(
            f"API call failed (attempt {retry_state.attempt_number}/{max_retries + 1}), "
            f"retrying in {retry_state.next_action.sleep:.1f}s: {retry_state.outcome.exception}"
        ),
        reraise=True
    )


class APIClient:
    """Base API client with rate limiting, retry logic, and error handling"""

    def __init__(self, base_url: str, api_key: Optional[str] = None, rate_limit: float = 1.0,
                 max_retries: int = 3, backoff_factor: float = 2.0, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.rate_limit = rate_limit
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout
        self.last_request_time = 0
        self._retry_decorator = create_retry_decorator(max_retries, backoff_factor)

    def _rate_limit_wait(self):
        """Enforce rate limiting between requests"""
        elapsed = time.time() - self.last_request_time
        if elapsed < (1.0 / self.rate_limit):
            wait_time = (1.0 / self.rate_limit) - elapsed
            logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
            time.sleep(wait_time)
        self.last_request_time = time.time()

    def _handle_response_errors(self, response: requests.Response, url: str) -> None:
        """Handle different types of HTTP errors and raise appropriate exceptions"""
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if response.status_code == 401:
                raise APIAuthenticationError(f"Authentication failed for {url}: {e}")
            elif response.status_code == 429:
                raise APIRateLimitError(f"Rate limit exceeded for {url}: {e}")
            elif response.status_code >= 500:
                raise APIError(f"Server error for {url}: {e}")
            else:
                raise APIError(f"HTTP error for {url}: {e}")
        except requests.exceptions.Timeout:
            raise APITimeoutError(f"Request timeout for {url}")
        except requests.exceptions.ConnectionError:
            raise APIError(f"Connection error for {url}")

    def _make_request(self, url: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict:
        """Make the actual HTTP request with error handling"""
        self._rate_limit_wait()

        _headers = headers or {}
        if self.api_key:
            _headers['Authorization'] = f'Bearer {self.api_key}'

        response = requests.get(url, params=params, headers=_headers, timeout=self.timeout)
        self._handle_response_errors(response, url)

        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise APIError(f"Invalid JSON response from {url}: {e}")

    @functools.lru_cache(maxsize=100)
    def get(self, endpoint: str, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Optional[Dict]:
        """Make GET request with retry logic and error handling"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        try:
            # Apply retry decorator to the actual request
            retried_request = self._retry_decorator(self._make_request)
            return retried_request(url, params, headers)
        except (APIError, requests.exceptions.RequestException) as e:
            logger.error(f"API request failed for {url}: {e}")
            return None


class ArtificialAnalysisClient(APIClient):
    """Client for Artificial Analysis API"""

    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 1.0,
                 max_retries: int = 3, backoff_factor: float = 2.0, timeout: int = 30):
        super().__init__(
            base_url="https://api.artificialanalysis.ai/v1",
            api_key=api_key,
            rate_limit=rate_limit,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            timeout=timeout
        )

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Fetch comprehensive model information"""
        # Try exact match first
        data = self.get(f"models/{model_name}")

        if not data:
            # Try fuzzy search
            all_models = self.get("models")
            if all_models:
                for model in all_models.get('models', []):
                    if model_name.lower() in model.get('name', '').lower():
                        return self.get(f"models/{model['id']}")

        return data

    def extract_benchmarks(self, model_data: Dict) -> Dict[str, Any]:
        """Extract benchmark scores from API response"""
        benchmarks = {}

        if not model_data:
            return benchmarks

        # Map API fields to our JSON structure
        benchmark_mapping = {
            'mmlu_pro': 'MMLU Pro',
            'gpqa': 'GPQA diamond',
            'math': 'MATH',
            'humaneval': 'HumanEval',
            'mbpp': 'MBPP',
            'aime': 'AIME',
            'livebench': 'LiveCodeBench',
            'scicode': 'SciCode',
            'intelligence_index': 'artificial_analysis'
        }

        scores = model_data.get('benchmarks', {})
        for api_key, json_key in benchmark_mapping.items():
            if api_key in scores:
                benchmarks[json_key] = scores[api_key]

        return benchmarks

    def extract_specs(self, model_data: Dict) -> ModelSpecs:
        """Extract model specifications"""
        if not model_data:
            return ModelSpecs()

        return ModelSpecs(
            input_price=model_data.get('pricing', {}).get('input_per_million'),
            output_price=model_data.get('pricing', {}).get('output_per_million'),
            context_window=model_data.get('context_window'),
            param_count=model_data.get('parameters'),
            architecture=model_data.get('architecture')
        )


class HuggingFaceClient(APIClient):
    """Client for Hugging Face API"""

    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 0.5,
                 max_retries: int = 3, backoff_factor: float = 2.0, timeout: int = 30):
        super().__init__(
            base_url="https://huggingface.co/api",
            api_key=api_key,
            rate_limit=rate_limit,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            timeout=timeout
        )

    def get_model_card(self, model_id: str) -> Optional[Dict]:
        """Fetch model card information"""
        return self.get(f"models/{model_id}")

    def extract_info(self, model_card: Dict) -> Dict[str, Any]:
        """Extract relevant info from model card"""
        info = {}

        if not model_card:
            return info

        # Extract from model card metadata
        metadata = model_card.get('cardData', {})

        # Try to get parameter count
        if 'model_size' in metadata:
            info['param_count'] = metadata['model_size']

        # Check for benchmark results in model card
        if 'model-index' in metadata:
            results = metadata['model-index'][0].get('results', [])
            for result in results:
                task = result.get('task', {}).get('type', '')
                metrics = result.get('metrics', [])

                # Map common benchmark names
                if 'mmlu' in task.lower():
                    for metric in metrics:
                        if metric.get('type') == 'accuracy':
                            info['MMLU'] = metric.get('value')

        return info


class LLMBenchmarkPipeline:
    """Main pipeline for filling LLM benchmark JSONs"""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline with configuration"""
        try:
            # Validate configuration
            validated_config = PipelineConfig(**config.model_dump() if hasattr(config, 'model_dump') else config.__dict__)

            self.config = validated_config
            self.aa_client = ArtificialAnalysisClient(
                api_key=validated_config.artificial_analysis_key,
                rate_limit=validated_config.rate_limit_aa,
                max_retries=validated_config.max_retries,
                backoff_factor=validated_config.retry_backoff_factor,
                timeout=validated_config.timeout
            )
            self.hf_client = HuggingFaceClient(
                api_key=validated_config.huggingface_key,
                rate_limit=validated_config.rate_limit_hf,
                max_retries=validated_config.max_retries,
                backoff_factor=validated_config.retry_backoff_factor,
                timeout=validated_config.timeout
            )

            logger.info("Pipeline initialized successfully")
            logger.info(f"AA API rate limit: {validated_config.rate_limit_aa} req/s")
            logger.info(f"HF API rate limit: {validated_config.rate_limit_hf} req/s")
            logger.info(f"Max retries: {validated_config.max_retries}")

        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    def load_template(self, filepath: str) -> Dict:
        """Load and validate JSON template"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate template structure
            template = ModelBenchmarkTemplate(**data)
            logger.info(f"Template loaded and validated from {filepath}")
            return template.model_dump()

        except FileNotFoundError:
            raise FileNotFoundError(f"Template file not found: {filepath}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in template file {filepath}: {e}")
        except ValidationError as e:
            raise ValueError(f"Template validation failed for {filepath}: {e}")

    def save_result(self, data: Dict, filepath: str):
        """Save filled JSON"""
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        logger.info(f"Saved result to {filepath}")

    def fill_model_data(self,
                       template: Dict,
                       model_info: ModelInfo) -> Dict:
        """Fill template with data from multiple sources"""

        # Validate model info
        validated_model = ModelInfo(**model_info.model_dump() if hasattr(model_info, 'model_dump') else model_info.__dict__)

        logger.info(f"Processing model: {validated_model.name}")

        # Start with template copy
        result = template.copy()

        # 1. Fetch from Artificial Analysis
        logger.info("Fetching from Artificial Analysis...")
        aa_data = self.aa_client.get_model_info(validated_model.name)

        if aa_data:
            # Fill benchmarks
            benchmarks = self.aa_client.extract_benchmarks(aa_data)
            for bench_name, score in benchmarks.items():
                if bench_name in result['dev_benchmarks']:
                    result['dev_benchmarks'][bench_name] = score
                elif bench_name == 'artificial_analysis':
                    result['entity_benchmarks']['artificial_analysis'] = score

            # Fill model specs
            specs = self.aa_client.extract_specs(aa_data)
            result['model_specs'].update(
                {k: v for k, v in asdict(specs).items() if v is not None}
            )

            logger.info(f"  ✓ Retrieved {len(benchmarks)} benchmarks from AA")
        else:
            logger.warning("  ✗ No data from Artificial Analysis")

        # 2. Fetch from Hugging Face (if model ID provided)
        if validated_model.hf_id:
            logger.info("Fetching from Hugging Face...")
            hf_data = self.hf_client.get_model_card(validated_model.hf_id)

            if hf_data:
                hf_info = self.hf_client.extract_info(hf_data)

                # Fill in missing data
                for key, value in hf_info.items():
                    if key in result['dev_benchmarks'] and result['dev_benchmarks'][key] is None:
                        result['dev_benchmarks'][key] = value
                    elif key == 'param_count' and result['model_specs']['param_count'] is None:
                        result['model_specs']['param_count'] = value

                logger.info("  ✓ Retrieved data from Hugging Face")
            else:
                logger.warning("  ✗ No data from Hugging Face")

        # 3. Calculate coverage (only count fields that were originally None and got filled)
        def count_filled_fields(original: Dict, current: Dict) -> int:
            """Count fields that were originally None/empty and now have values"""
            filled = 0
            for key in original.keys():
                if original[key] is None and current.get(key) is not None:
                    filled += 1
            return filled

        filled_entity = count_filled_fields(template['entity_benchmarks'], result['entity_benchmarks'])
        filled_dev = count_filled_fields(template['dev_benchmarks'], result['dev_benchmarks'])
        filled_community = count_filled_fields(template['community_score'], result['community_score'])
        filled_specs = count_filled_fields(template['model_specs'], result['model_specs'])

        total_fillable_fields = (
            sum(1 for v in template['entity_benchmarks'].values() if v is None) +
            sum(1 for v in template['dev_benchmarks'].values() if v is None) +
            sum(1 for v in template['community_score'].values() if v is None) +
            sum(1 for v in template['model_specs'].values() if v is None)
        )

        total_filled_fields = filled_entity + filled_dev + filled_community + filled_specs

        coverage = (total_filled_fields / total_fillable_fields * 100) if total_fillable_fields > 0 else 100.0
        logger.info(f"  Coverage: {coverage:.1f}% ({total_filled_fields}/{total_fillable_fields} fillable fields)")

        return result

    def process_batch(self, models: List[Union[Dict, ModelInfo]]) -> List[Dict]:
        """Process multiple models in batch"""

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        # Load and validate template
        template = self.load_template(self.config.template_path)

        results = []
        successful_models = 0

        for i, model_info in enumerate(models, 1):
            try:
                # Validate model info
                if isinstance(model_info, dict):
                    validated_model = ModelInfo(**model_info)
                else:
                    validated_model = model_info

                logger.info(f"\n[{i}/{len(models)}] Processing {validated_model.name}")

                filled_data = self.fill_model_data(template=template, model_info=validated_model)

                # Create safe filename
                safe_name = validated_model.name.replace('/', '_').replace('\\', '_')
                output_path = Path(self.config.output_dir) / f"{safe_name}_filled.json"
                self.save_result(filled_data, str(output_path))

                results.append({
                    'model': validated_model.name,
                    'status': 'success',
                    'output': str(output_path)
                })
                successful_models += 1

            except ValidationError as e:
                error_msg = f"Model validation failed: {e}"
                logger.error(f"Error processing {model_info.get('name', 'unknown')}: {error_msg}")
                results.append({
                    'model': model_info.get('name', 'unknown') if isinstance(model_info, dict) else str(model_info),
                    'status': 'validation_error',
                    'error': error_msg
                })
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                logger.error(f"Error processing {model_info.get('name', 'unknown') if isinstance(model_info, dict) else str(model_info)}: {error_msg}")
                results.append({
                    'model': model_info.get('name', 'unknown') if isinstance(model_info, dict) else str(model_info),
                    'status': 'error',
                    'error': error_msg
                })

                # Continue processing if configured to do so
                if not self.config.continue_on_error:
                    logger.error("Stopping batch processing due to error (continue_on_error=False)")
                    break

        # Save batch summary
        summary_path = Path(self.config.output_dir) / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nBatch processing complete. {successful_models}/{len(models)} models successful.")
        logger.info(f"Summary saved to {summary_path}")
        return results


def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML or JSON file"""
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            import yaml
            config_data = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config_data = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")

    return config_data


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="LLM Benchmark Data Pipeline - Fill model benchmark JSONs using multiple API sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (recommended)
  python llm_benchmark_pipeline.py launch

  # Basic usage with environment variables
  python llm_benchmark_pipeline.py --template GLM-4.6.json --models models.json

  # With config file
  python llm_benchmark_pipeline.py --config pipeline_config.yaml

  # Override specific settings
  python llm_benchmark_pipeline.py --template GLM-4.6.json --output-dir ./results --max-retries 5
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Launch interactive command
    subparsers.add_parser('launch',
                         help='Launch interactive pipeline mode')

    # Config file
    parser.add_argument('--config', '-c',
                       help='Configuration file (YAML or JSON)')

    # Required arguments (can be in config file)
    parser.add_argument('--template', '-t',
                       help='Path to template JSON file')
    parser.add_argument('--models', '-m',
                       help='Path to models JSON file or inline JSON string')

    # Optional overrides
    parser.add_argument('--output-dir', '-o', default='./filled_models',
                       help='Output directory for results (default: ./filled_models)')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum retry attempts for API calls (default: 3)')
    parser.add_argument('--rate-limit-aa', type=float, default=1.0,
                       help='Rate limit for AA API (requests/second, default: 1.0)')
    parser.add_argument('--rate-limit-hf', type=float, default=0.5,
                       help='Rate limit for HF API (requests/second, default: 0.5)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Request timeout in seconds (default: 30)')
    parser.add_argument('--continue-on-error', action='store_true', default=True,
                       help='Continue processing other models if one fails (default: True)')
    parser.add_argument('--no-continue-on-error', action='store_true',
                       help='Stop processing on first error')

    # API keys (can use environment variables)
    parser.add_argument('--aa-key',
                       help='Artificial Analysis API key (or set ARTIFICIAL_ANALYSIS_API_KEY env var)')
    parser.add_argument('--hf-key',
                       help='Hugging Face API key (or set HUGGINGFACE_API_KEY env var)')

    # Logging
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Set logging level (default: INFO)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging (same as --log-level DEBUG)')

    return parser


def load_models(models_path_or_json: str) -> List[Dict]:
    """Load models from file or parse JSON string"""
    try:
        # Try to parse as JSON first
        models = json.loads(models_path_or_json)
        if isinstance(models, list):
            return models
    except (json.JSONDecodeError, TypeError):
        pass

    # Try to load from file
    if Path(models_path_or_json).exists():
        with open(models_path_or_json, 'r', encoding='utf-8') as f:
            models = json.load(f)
        if isinstance(models, list):
            return models

    raise ValueError(f"Could not load models from: {models_path_or_json}")


def scan_json_folder(folder: str) -> List[Path]:
    """Scan folder for JSON files and return sorted list of paths"""
    folder_path = Path(folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")
    if not folder_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {folder}")

    json_files = sorted([f for f in folder_path.iterdir() if f.suffix.lower() == '.json'])
    return json_files


def detect_models_in_file(file_path: Path) -> List[str]:
    """Detect model names in a JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Check if it's a model benchmark template
            if any(key in data for key in ['entity_benchmarks', 'dev_benchmarks', 'model_specs']):
                # Try to get model name from model_specs
                if 'model_specs' in data and isinstance(data['model_specs'], dict):
                    if 'model_name' in data['model_specs']:
                        return [data['model_specs']['model_name']]
                # Fall back to filename
                return [file_path.stem]

            # Check if it's a model descriptor with 'name' field
            if 'name' in data and isinstance(data['name'], str):
                return [data['name']]

        elif isinstance(data, list):
            # List of model descriptors
            names = []
            for item in data:
                if isinstance(item, dict) and 'name' in item:
                    names.append(item['name'])
            return names

        # Default to filename
        return [file_path.stem]

    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Could not read {file_path}: {e}")
        return [file_path.stem]


def prompt_for_api_keys() -> tuple[str | None, str | None]:
    """Prompt user for API keys if not in environment"""
    aa_key = os.getenv('ARTIFICIAL_ANALYSIS_API_KEY')
    if not aa_key:
        aa_key = input("Artificial Analysis API key (or ENTER to skip): ").strip()
        aa_key = aa_key if aa_key else None

    hf_key = os.getenv('HUGGINGFACE_API_KEY')
    if not hf_key:
        hf_key = input("Hugging Face API key (or ENTER to skip): ").strip()
        hf_key = hf_key if hf_key else None

    return aa_key, hf_key


def launch_interactive():
    """Launch interactive pipeline mode"""
    print("🤖 LLM Benchmark Pipeline - Interactive Mode")
    print("=" * 50)

    # 1. Prompt for API keys
    print("\n🔑 API Keys")
    aa_key, hf_key = prompt_for_api_keys()

    if not aa_key:
        print("⚠️  Warning: No Artificial Analysis API key provided")
    if not hf_key:
        print("⚠️  Warning: No Hugging Face API key provided")

    # 2. Prompt for input folder
    print("\n📁 Input Folder")
    while True:
        input_folder = input("Input folder containing model JSONs (default: ./): ").strip()
        input_folder = input_folder if input_folder else "."

        try:
            json_files = scan_json_folder(input_folder)
            break
        except (FileNotFoundError, NotADirectoryError) as e:
            print(f"❌ Error: {e}")
            continue

    if not json_files:
        print("❌ No JSON files found in the specified folder.")
        return

    # 3. Detect and list models
    print(f"\n🔍 Found {len(json_files)} JSON file(s)")
    print("\n📋 Detected models:")

    detected_models = []
    for i, file_path in enumerate(json_files, 1):
        model_names = detect_models_in_file(file_path)
        detected_models.append({
            'file': file_path,
            'names': model_names,
            'index': i
        })
        print(f"  {i}. {file_path.name} -> {', '.join(model_names)}")

    # 4. Confirm processing
    print("\n✅ Confirmation")
    confirm = input("Process all detected files? [Y/n]: ").strip().lower()
    if confirm.startswith('n'):
        # Allow user to select specific files
        selection = input("Enter comma-separated numbers to process (e.g., 1,3,5): ").strip()
        if selection:
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                detected_models = [detected_models[i] for i in indices if 0 <= i < len(detected_models)]
            except (ValueError, IndexError):
                print("❌ Invalid selection, processing all files")
        else:
            print("❌ No selection made, exiting")
            return

    if not detected_models:
        print("❌ No files selected for processing")
        return

    # 5. Prompt for output folder
    print("\n📂 Output Folder")
    output_dir = input("Output folder for results (default: ./filled_models): ").strip()
    output_dir = output_dir if output_dir else "./filled_models"

    # 6. Ask for verbose mode
    verbose = input("Enable verbose updates during processing? [y/N]: ").strip().lower().startswith('y')

    # Set logging level
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # 7. Initialize pipeline
    print(f"\n🚀 Starting processing of {len(detected_models)} file(s)...")

    config = PipelineConfig(
        template_path="",  # Not used in individual file processing
        output_dir=output_dir,
        artificial_analysis_key=aa_key,
        huggingface_key=hf_key,
        continue_on_error=True  # Always continue in interactive mode
    )

    pipeline = LLMBenchmarkPipeline(config)

    # 8. Process each file individually
    results = []
    successful = 0

    for i, model_info in enumerate(detected_models, 1):
        file_path = model_info['file']
        model_names = model_info['names']

        print(f"\n[{i}/{len(detected_models)}] Processing: {file_path.name}")

        try:
            # Load template from file
            template = pipeline.load_template(str(file_path))

            # Extract model name for processing
            model_name = model_names[0] if model_names else file_path.stem

            # Try to extract HF ID from the template if it exists
            hf_id = None
            if isinstance(template, dict) and 'hf_id' in template:
                hf_id = template['hf_id']

            # Create validated model info
            model_info_obj = ModelInfo(name=model_name, hf_id=hf_id)

            # Process the model
            if verbose:
                print(f"   📝 Processing model: {model_name}")
            filled_data = pipeline.fill_model_data(template, model_info_obj)

            # Save result
            safe_name = file_path.stem.replace('/', '_').replace('\\', '_')
            output_path = Path(output_dir) / f"{safe_name}_filled.json"
            pipeline.save_result(filled_data, str(output_path))

            results.append({
                'file': str(file_path),
                'model': model_name,
                'status': 'success',
                'output': str(output_path)
            })
            successful += 1

            if verbose:
                print(f"   ✅ Saved to: {output_path}")

        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Error: {error_msg}")

            results.append({
                'file': str(file_path),
                'model': model_names[0] if model_names else file_path.stem,
                'status': 'error',
                'error': error_msg
            })

    # 9. Rich recap
    print("\n" + "=" * 60)
    print("📊 PROCESSING COMPLETE")
    print("=" * 60)

    print(f"📁 Total files processed: {len(results)}")
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {len(results) - successful}")

    if successful > 0:
        print(f"\n📂 Results saved to: {output_dir}")
        print("\n📋 Successful files:")
        for result in results:
            if result['status'] == 'success':
                print(f"  ✅ {Path(result['file']).name} -> {Path(result['output']).name}")

    if len(results) > successful:
        print("\n❌ Failed files:")
        for result in results:
            if result['status'] == 'error':
                print(f"  ❌ {Path(result['file']).name}: {result.get('error', 'Unknown error')}")

    print("\n🎉 Interactive processing complete!")
    print("=" * 60)


def main():
    """Main entry point with configuration management"""
    parser = create_parser()
    args = parser.parse_args()

    # Handle interactive launch command
    if args.command == 'launch':
        try:
            launch_interactive()
            exit(0)
        except KeyboardInterrupt:
            print("\n\n❌ Operation cancelled by user")
            exit(1)
        except Exception as e:
            logger.error(f"Interactive mode failed: {e}")
            exit(1)

    # Set logging level for non-interactive modes
    log_level = 'DEBUG' if args.verbose else args.log_level
    logging.getLogger().setLevel(getattr(logging, log_level))

    try:
        # Load configuration
        config_data = {}

        # Load from config file if provided
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config_data = load_config_from_file(args.config)

        # Override with command line arguments
        if args.template:
            config_data['template_path'] = args.template
        if args.output_dir:
            config_data['output_dir'] = args.output_dir
        if args.max_retries is not None:
            config_data['max_retries'] = args.max_retries
        if args.rate_limit_aa is not None:
            config_data['rate_limit_aa'] = args.rate_limit_aa
        if args.rate_limit_hf is not None:
            config_data['rate_limit_hf'] = args.rate_limit_hf
        if args.timeout is not None:
            config_data['timeout'] = args.timeout
        if args.no_continue_on_error:
            config_data['continue_on_error'] = False

        # Load API keys from args or environment
        if args.aa_key:
            config_data['artificial_analysis_key'] = args.aa_key
        elif 'ARTIFICIAL_ANALYSIS_API_KEY' in os.environ:
            config_data['artificial_analysis_key'] = os.environ['ARTIFICIAL_ANALYSIS_API_KEY']

        if args.hf_key:
            config_data['huggingface_key'] = args.hf_key
        elif 'HUGGINGFACE_API_KEY' in os.environ:
            config_data['huggingface_key'] = os.environ['HUGGINGFACE_API_KEY']

        # Validate required parameters
        if not args.models and 'models' not in config_data:
            parser.error("--models is required (or specify in config file)")
        if not args.template and 'template_path' not in config_data:
            parser.error("--template is required (or specify in config file)")

        # Load models
        if args.models:
            models = load_models(args.models)
        else:
            models = config_data['models']

        # Create pipeline config
        pipeline_config = PipelineConfig(**config_data)

        # Initialize and run pipeline
        pipeline = LLMBenchmarkPipeline(pipeline_config)
        results = pipeline.process_batch(models)

        # Print summary
        print("\n" + "="*60)
        print("BATCH PROCESSING SUMMARY")
        print("="*60)

        success_count = sum(1 for r in results if r['status'] == 'success')
        error_count = sum(1 for r in results if r['status'] in ['error', 'validation_error'])

        print(f"Total models: {len(results)}")
        print(f"Successful: {success_count}")
        print(f"Errors: {error_count}")
        print()

        for result in results:
            status_symbol = "✓" if result['status'] == 'success' else "✗"
            status_info = result['status']
            if result['status'] != 'success':
                status_info += f" - {result.get('error', '')}"
            print(f"{status_symbol} {result['model']}: {status_info}")

        # Exit with appropriate code
        exit(0 if error_count == 0 else 1)

    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
