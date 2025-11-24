#!/usr/bin/env python3

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
LLM Benchmark Data Pipeline
=============================
Automated pipeline to fill model benchmark JSONs using multiple API sources.

Supports:
- Artificial Analysis API (benchmark scores, pricing, specs)
- Hugging Face API (model cards, parameters, leaderboard data)
- Batch processing of multiple model JSONs

Requirements:
    pip install requests pandas tqdm python-dotenv pydantic pyyaml tenacity aiohttp

Environment variables (.env file):
    ARTIFICIAL_ANALYSIS_API_KEY=your_key_here
    HUGGINGFACE_API_KEY=your_key_here  # optional, for rate limits
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import csv
import json
import os
import time
import argparse
import re
import copy
import asyncio
from typing import Dict, Any, Optional, List, Union, Sequence, Tuple, Pattern
from pathlib import Path
import aiohttp
import requests
from dataclasses import dataclass, asdict
import logging
from pydantic import BaseModel, ValidationError, field_validator, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _load_moe_override_patterns() -> Tuple[List[Pattern[str]], List[Pattern[str]]]:
    script_dir = Path(__file__).parent
    overrides_path = script_dir / "moe_model_overrides.json"
    if not overrides_path.exists():
        return [], []

    try:
        with open(overrides_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to load MoE overrides from %s: %s", overrides_path, exc)
        return [], []

    def _compile(patterns: List[str]) -> List[Pattern[str]]:
        compiled: List[Pattern[str]] = []
        for pattern in patterns:
            try:
                compiled.append(re.compile(pattern, re.IGNORECASE))
            except re.error as regex_exc:
                logger.warning(
                    "Invalid MoE override regex '%s' in %s: %s",
                    pattern,
                    overrides_path,
                    regex_exc,
                )
        return compiled

    include = _compile(data.get('moe_patterns', []))
    exclude = _compile(data.get('moe_exclude_patterns', []))
    return include, exclude


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
    cache_ttl: int = Field(3600, ge=0, description="Cache TTL in seconds for API responses")
    continue_on_error: bool = Field(True, description="Continue processing other models if one fails")

    @field_validator('template_path', 'output_dir')
    @classmethod
    def validate_paths(cls, v):
        path = Path(v)
        if 'template_path' in cls.model_fields and not path.exists():
            raise ValueError(f'Path does not exist: {v}')
        return str(path)


MOE_NAME_REGEXES = [
    re.compile(r"\b\d+\s*b\s*-\s*\d+\s*x\s*\d+\s*b\b", re.IGNORECASE),
    re.compile(r"\b\d+(?:\.\d+)?\s*b[\s-]+[a-z]+\d+(?:x\d+)?\s*b\b", re.IGNORECASE),
    re.compile(r"\b\d+\s*x\s*\d+\s*b\b", re.IGNORECASE),
]


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
            aiohttp.ClientError,
            asyncio.TimeoutError,
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
        self._request_lock = asyncio.Lock()

    async def _rate_limit_wait(self):
        """Enforce rate limiting between requests"""
        async with self._request_lock:
            elapsed = time.time() - self.last_request_time
            if elapsed < (1.0 / self.rate_limit):
                wait_time = (1.0 / self.rate_limit) - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self.last_request_time = time.time()

    def _handle_response_errors(self, status: int, url: str, text: str = "") -> None:
        """Handle different types of HTTP errors and raise appropriate exceptions"""
        if 200 <= status < 300:
            return
            
        if status == 401:
            raise APIAuthenticationError(f"Authentication failed for {url}")
        elif status == 429:
            raise APIRateLimitError(f"Rate limit exceeded for {url}")
        elif status >= 500:
            raise APIError(f"Server error {status} for {url}: {text}")
        else:
            raise APIError(f"HTTP error {status} for {url}: {text}")

    async def _make_request(self, url: str, session: aiohttp.ClientSession, params: Optional[Dict] = None, headers: Optional[Dict] = None) -> Dict:
        """Make the actual HTTP request with error handling"""
        await self._rate_limit_wait()

        _headers = headers or {}
        if self.api_key:
            _headers['Authorization'] = f'Bearer {self.api_key}'

        try:
            async with session.get(url, params=params, headers=_headers, timeout=self.timeout) as response:
                text = await response.text()
                self._handle_response_errors(response.status, url, text)
                
                try:
                    return json.loads(text)
                except json.JSONDecodeError as e:
                    raise APIError(f"Invalid JSON response from {url}: {e}")
                    
        except asyncio.TimeoutError:
            raise APITimeoutError(f"Request timeout for {url}")
        except aiohttp.ClientError as e:
            raise APIError(f"Connection error for {url}: {e}")

    async def get(self, endpoint: str, params: Optional[Dict] = None, headers: Optional[Dict] = None, session: Optional[aiohttp.ClientSession] = None) -> Optional[Dict]:
        """Make GET request with retry logic and error handling"""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        local_session = False
        if session is None:
            session = aiohttp.ClientSession()
            local_session = True

        try:
            # Apply retry decorator to the actual request
            # Note: we pass session as a keyword argument to match the signature if needed, 
            # but since _make_request is bound, we just call it.
            # However, tenacity decorates the wrapper.
            
            @self._retry_decorator
            async def _do_request():
                return await self._make_request(url, session, params, headers)

            return await _do_request()
            
        except (APIError, aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"API request failed for {url}: {e}")
            return None
        finally:
            if local_session:
                await session.close()


def normalize_name(name: str) -> str:
    """Normalize model name for fuzzy matching (remove non-alphanumeric, lowercase)"""
    return re.sub(r'[^a-z0-9]', '', name.lower())


class ArtificialAnalysisClient(APIClient):
    """Client for Artificial Analysis API (v2 data endpoints)."""

    LLMS_ENDPOINT = "https://artificialanalysis.ai/api/v2/data/llms"

    def __init__(self, api_key: Optional[str] = None, rate_limit: float = 1.0,
                 max_retries: int = 3, backoff_factor: float = 2.0, timeout: int = 30,
                 model_mapping: Optional[Dict[str, str]] = None,
                 cache_dir: Optional[str] = None, cache_ttl: int = 3600):
        super().__init__(
            base_url=self.LLMS_ENDPOINT,
            api_key=api_key,
            rate_limit=rate_limit,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            timeout=timeout,
        )
        self.model_mapping = model_mapping or {}
        self._model_cache: Optional[List[Dict[str, Any]]] = None
        
        # Caching configuration
        self.cache_ttl = cache_ttl
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / ".cache"
        self.cache_file = self.cache_dir / "aa_models_cache.json"

    def _load_cache(self) -> Optional[List[Dict[str, Any]]]:
        """Load models from persistent cache if valid"""
        if not self.cache_file.exists():
            return None
            
        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            timestamp = data.get('timestamp', 0)
            age = time.time() - timestamp
            
            if age > self.cache_ttl:
                logger.debug(f"Cache expired (age: {age:.0f}s > ttl: {self.cache_ttl}s)")
                return None
                
            models = data.get('models', [])
            if models:
                logger.info(f"Loaded {len(models)} models from cache ({age:.0f}s old)")
                return models
                
        except Exception as e:
            logger.warning(f"Failed to read cache: {e}")
            return None

    def _save_cache(self, models: List[Dict[str, Any]]) -> None:
        """Save models to persistent cache"""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            data = {
                'timestamp': time.time(),
                'models': models
            }
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f)
            logger.debug(f"Saved {len(models)} models to cache")
        except Exception as e:
            logger.warning(f"Failed to write cache: {e}")

    def _aa_headers(self) -> Dict[str, str]:
        if not self.api_key:
            raise APIAuthenticationError("Artificial Analysis API key is required.")
        return {"x-api-key": self.api_key}

    async def _fetch_llm_payload(self, endpoint: str = "models", session: Optional[aiohttp.ClientSession] = None) -> Dict[str, Any]:
        """Invoke the documented v2 LLMS endpoint."""
        # Note: _rate_limit_wait is handled in _make_request inside get()
        # We construct the URL relative to base_url manually here because APIClient.get expects just the endpoint
        # But since APIClient.get calls _make_request which does rate limiting, we can just use self.get
        
        # However, self.get wraps _make_request. 
        
        response = await self.get(endpoint, headers=self._aa_headers(), session=session)
        if response is None:
             raise APIError(f"Failed to fetch from {endpoint}")
        return response

    async def list_models(self, session: Optional[aiohttp.ClientSession] = None) -> List[Dict[str, Any]]:
        if self._model_cache is not None:
            return self._model_cache
            
        # Try loading from persistent cache
        cached_models = self._load_cache()
        if cached_models:
            self._model_cache = cached_models
            return cached_models

        try:
            payload = await self._fetch_llm_payload("models", session=session)
            models = payload.get("data", [])
            if not isinstance(models, list):
                logger.warning("Unexpected response shape for AA models endpoint")
                models = []
                
            # Update memory and persistent cache
            self._model_cache = models
            self._save_cache(models)
            
            return models
        except APIError as e:
            logger.error(f"Failed to list AA models: {e}")
            return []

    async def get_model_info(self, model_name: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[Dict]:
        """Fetch comprehensive model information"""
        # 1. Check explicit mapping first
        models = await self.list_models(session=session)
        if model_name in self.model_mapping:
            mapped_value = self.model_mapping[model_name]
            logger.info("Using mapped ID for %s -> %s", model_name, mapped_value)

            def _flatten_tokens(value: Any) -> List[str]:
                if isinstance(value, dict):
                    return [
                        value.get("id"),
                        value.get("slug"),
                        value.get("name"),
                    ]
                if isinstance(value, (list, tuple, set)):
                    tokens: List[str] = []
                    for item in value:
                        tokens.extend(_flatten_tokens(item))
                    return tokens
                return [value]

            target_tokens = [token for token in _flatten_tokens(mapped_value) if token]
            normalized_targets = [normalize_name(token) for token in target_tokens if token]

            for entry in models:
                entry_tokens = [entry.get("id"), entry.get("slug"), entry.get("name")]
                entry_norm = [normalize_name(token) for token in entry_tokens if token]

                if any(token == entry_token for token in target_tokens for entry_token in entry_tokens if token and entry_token):
                    return entry
                if any(target == entry_token for target in normalized_targets for entry_token in entry_norm if target and entry_token):
                    return entry

            logger.warning("Mapped target %s not found in AA catalog", mapped_value)

        normalized_target = normalize_name(model_name)

        if logger.level <= logging.DEBUG:
            logger.debug("Available AA models: %s", [m.get("name") for m in models])

        def _matches(entry: Dict[str, Any]) -> bool:
            candidates = [entry.get("name", ""), entry.get("slug", ""), entry.get("id", "")]
            return any(normalized_target == normalize_name(candidate) for candidate in candidates)

        for entry in models:
            if _matches(entry):
                return entry

        for entry in models:
            norm_name = normalize_name(entry.get("name", ""))
            if (normalized_target in norm_name or norm_name in normalized_target) and len(norm_name) > 5:
                logger.info("Partial AA match %s -> %s", model_name, entry.get("name"))
                return entry

        return None

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
            'math_500': 'MATH',
            'humaneval': 'HumanEval',
            'mbpp': 'MBPP',
            'aime': 'AIME',
            'aime_25': 'AIME',
            'livecodebench': 'LiveCodeBench',
            'livebench': 'LiveCodeBench',
            'scicode': 'SciCode',
            'hle': "Humanity's Last Exam",
            'artificial_analysis_intelligence_index': 'artificial_analysis',
            'intelligence_index': 'artificial_analysis'
        }

        scores = model_data.get('evaluations', {})
        for api_key, json_key in benchmark_mapping.items():
            if api_key in scores:
                benchmarks[json_key] = scores[api_key]

        return benchmarks

    def extract_specs(self, model_data: Dict) -> ModelSpecs:
        """Extract model specifications"""
        if not model_data:
            return ModelSpecs()

        pricing = model_data.get('pricing', {})
        creator = model_data.get('model_creator', {})
        return ModelSpecs(
            input_price=pricing.get('price_1m_input_tokens'),
            output_price=pricing.get('price_1m_output_tokens'),
            context_window=model_data.get('context_window'),
            param_count=model_data.get('parameters'),
            architecture=creator.get('name') or model_data.get('slug'),
        )


class BaseLeaderboardClient:
    """Shared helpers for leaderboard clients using cached JSON payloads."""

    def __init__(self, cache_file: Path, cache_ttl: int = 3600):
        self.cache_file = cache_file
        self.cache_ttl = cache_ttl
        self._cache_timestamp: float = 0.0
        self._entries: Optional[List[Dict[str, Any]]] = None

    def _load_cache(self) -> None:
        if not self.cache_file.exists():
            return
        try:
            with open(self.cache_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            timestamp = payload.get("timestamp", 0)
            if time.time() - timestamp > self.cache_ttl:
                return
            entries = payload.get("entries")
            if isinstance(entries, list):
                self._entries = entries
                self._cache_timestamp = timestamp
                logger.info("Loaded cache %s (%d entries)", self.cache_file.name, len(entries))
        except Exception as exc:
            logger.warning("Failed to load cache %s: %s", self.cache_file, exc)

    def _save_cache(self) -> None:
        if not self._entries:
            return
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as fh:
                json.dump({"timestamp": self._cache_timestamp, "entries": self._entries}, fh)
        except Exception as exc:
            logger.warning("Failed to save cache %s: %s", self.cache_file, exc)

    def _ensure_entries_loaded(self) -> bool:
        if self._entries and (time.time() - self._cache_timestamp) <= self.cache_ttl:
            return True
        if self._entries is None:
            self._load_cache()
            if self._entries:
                return True
        return False


class UGILeaderboardClient(BaseLeaderboardClient):
    """Fetch and cache UGI leaderboard scores from the public HF CSV."""

    CSV_URL = "https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard/resolve/main/ugi-leaderboard-data.csv"
    MODEL_COLUMN = "author/model_name"
    SCORE_COLUMNS = ("UGI 🏆", "UGI", "UGI Score")

    def __init__(self, cache_ttl: int = 3600, cache_dir: Optional[str] = None,
                 aliases: Optional[Dict[str, Union[str, List[str]]]] = None):
        cache_dir_path = Path(cache_dir) if cache_dir else Path(__file__).parent / ".cache"
        super().__init__(cache_file=cache_dir_path / "ugi_leaderboard_cache.json", cache_ttl=cache_ttl)
        self.cache_dir = cache_dir_path
        self._rows: Optional[List[Dict[str, Any]]] = None
        self.alias_map: Dict[str, List[str]] = {}
        if aliases:
            normalized: Dict[str, List[str]] = {}
            for key, values in aliases.items():
                norm_key = normalize_name(key)
                if not norm_key:
                    continue
                if isinstance(values, str):
                    values = [values]
                elif not isinstance(values, (list, tuple, set)):
                    continue
                normalized[norm_key] = [normalize_name(v) for v in values if isinstance(v, str)]
            self.alias_map = normalized

    async def _fetch_rows(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        if self._rows and (time.time() - self._cache_timestamp) <= self.cache_ttl:
            return

        if self._rows is None:
            self._ensure_entries_loaded()
            if self._rows:
                return

        local_session = False
        if session is None:
            session = aiohttp.ClientSession()
            local_session = True

        try:
            async with session.get(self.CSV_URL, timeout=60) as response:
                if response.status != 200:
                    logger.error("Failed to download UGI leaderboard: HTTP %s", response.status)
                    return
                text = await response.text()
        except Exception as exc:
            logger.error("Error downloading UGI leaderboard: %s", exc)
            return
        finally:
            if local_session:
                await session.close()

        reader = csv.DictReader(text.splitlines())
        if reader.fieldnames:
            reader.fieldnames = [name.lstrip("\ufeff") if isinstance(name, str) else name for name in reader.fieldnames]
        self._rows = [row for row in reader]
        self._cache_timestamp = time.time()
        self._entries = self._rows
        self._save_cache()
        logger.info("Fetched %d UGI leaderboard rows", len(self._rows))

    async def get_score(self, model_name: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[float]:
        if not model_name:
            return None

        await self._fetch_rows(session=session)
        if not self._rows:
            return None

        norm_target = normalize_name(model_name)
        target_norms = {norm_target}
        if norm_target in self.alias_map:
            target_norms.update(self.alias_map[norm_target])
        for row in self._rows:
            raw_model = row.get(self.MODEL_COLUMN, "")
            if not raw_model:
                continue
            candidates = [raw_model]
            if "/" in raw_model:
                candidates.append(raw_model.split("/")[-1])

            row_norms = {normalize_name(candidate) for candidate in candidates if candidate}
            if target_norms & row_norms:
                score_raw = None
                for column in self.SCORE_COLUMNS:
                    score_raw = row.get(column)
                    if score_raw:
                        break
                if not score_raw:
                    return None
                try:
                    return float(score_raw)
                except (TypeError, ValueError):
                    logger.warning("Invalid UGI score '%s' for model %s", score_raw, raw_model)
                    return None

        logger.debug("UGI leaderboard entry not found for %s", model_name)
        return None


class OpenVLMLeaderboardClient(BaseLeaderboardClient):
    """Client for Open VLM leaderboard JSON feed."""

    JSON_URL = "http://opencompass.openxlab.space/assets/OpenVLM.json"
    BENCHMARK_MAPPINGS: Dict[str, Tuple[str, str]] = {
        'MMMU': ('MMMU_VAL', 'Overall'),
        'Mathvista': ('MathVista', 'Overall'),
        'AI2D': ('AI2D', 'Overall'),
    }

    def __init__(self, cache_ttl: int = 3600, cache_dir: Optional[str] = None,
                 aliases: Optional[Dict[str, Union[str, List[str]]]] = None):
        cache_dir_path = Path(cache_dir) if cache_dir else Path(__file__).parent / ".cache"
        super().__init__(cache_file=cache_dir_path / "open_vlm_cache.json", cache_ttl=cache_ttl)
        self.cache_dir = cache_dir_path
        self._payload: Optional[Dict[str, Any]] = None
        self.alias_map: Dict[str, List[str]] = {}
        if aliases:
            normalized: Dict[str, List[str]] = {}
            for key, values in aliases.items():
                norm_key = normalize_name(key)
                if not norm_key:
                    continue
                if isinstance(values, str):
                    values = [values]
                elif not isinstance(values, (list, tuple, set)):
                    continue
                normalized[norm_key] = [normalize_name(v) for v in values if isinstance(v, str)]
            self.alias_map = normalized

    async def _fetch_payload(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        if self._payload and (time.time() - self._cache_timestamp) <= self.cache_ttl:
            return

        if self._payload is None:
            if self._ensure_entries_loaded():
                self._payload = {"results": {entry["model_name"]: entry for entry in self._entries}}
                return

        local_session = False
        if session is None:
            session = aiohttp.ClientSession()
            local_session = True

        try:
            async with session.get(self.JSON_URL, timeout=60) as response:
                if response.status != 200:
                    logger.error("Failed to download Open VLM leaderboard: HTTP %s", response.status)
                    return
                text = await response.text()
        except Exception as exc:
            logger.error("Error downloading Open VLM leaderboard: %s", exc)
            return
        finally:
            if local_session:
                await session.close()

        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.error("Invalid JSON from Open VLM leaderboard: %s", exc)
            return

        results = payload.get("results")
        if not isinstance(results, dict):
            logger.warning("Unexpected Open VLM payload shape")
            return

        self._payload = payload
        self._entries = [
            {"model_name": name, **data}
            for name, data in results.items()
        ]
        self._cache_timestamp = time.time()
        self._save_cache()
        logger.info("Fetched Open VLM leaderboard payload (%d entries)", len(self._entries))

    async def _find_result_entry(self, model_name: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[Dict[str, Any]]:
        if not model_name:
            return None

        await self._fetch_payload(session=session)
        if not self._payload:
            return None

        results = self._payload.get("results", {}) or {}
        norm_target = normalize_name(model_name)
        target_norms = {norm_target}
        if norm_target in self.alias_map:
            target_norms.update(self.alias_map[norm_target])

        for name, entry in results.items():
            row_norms = {normalize_name(name)}
            method_field = entry.get("META", {}).get("Method")
            if isinstance(method_field, list) and method_field:
                method_norm = normalize_name(method_field[0])
                if method_norm:
                    row_norms.add(method_norm)
            elif isinstance(method_field, str):
                method_norm = normalize_name(method_field)
                if method_norm:
                    row_norms.add(method_norm)

            if target_norms & {norm for norm in row_norms if norm}:
                return entry

        logger.debug("Open VLM entry not found for %s", model_name)
        return None

    async def get_score(self, model_name: str, session: Optional[aiohttp.ClientSession] = None,
                        score_key: str = "MMBench_TEST_EN", sub_key: str = "Overall") -> Optional[float]:
        entry = await self._find_result_entry(model_name, session=session)
        if not entry:
            return None

        section = entry.get(score_key)
        if isinstance(section, dict):
            score_val = section.get(sub_key)
            if isinstance(score_val, (int, float)):
                return float(score_val)
        return None

    async def get_benchmark_scores(
        self,
        model_name: str,
        session: Optional[aiohttp.ClientSession] = None,
        benchmark_map: Optional[Dict[str, Tuple[str, str]]] = None,
    ) -> Dict[str, float]:
        mapping = benchmark_map or self.BENCHMARK_MAPPINGS
        if not mapping:
            return {}

        entry = await self._find_result_entry(model_name, session=session)
        if not entry:
            return {}

        results: Dict[str, float] = {}
        for output_name, (score_key, sub_key) in mapping.items():
            section = entry.get(score_key)
            if not isinstance(section, dict):
                continue
            score_val = section.get(sub_key)
            if isinstance(score_val, (int, float)):
                results[output_name] = float(score_val)
        return results


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

    async def get_model_card(self, model_id: str, session: Optional[aiohttp.ClientSession] = None) -> Optional[Dict]:
        """Fetch model card information"""
        return await self.get(f"models/{model_id}", session=session)

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
            
            # Load model mapping if exists
            model_mapping = {}
            script_dir = Path(__file__).parent
            mapping_path = script_dir / "model_mapping.json"
            if mapping_path.exists():
                try:
                    with open(mapping_path, 'r', encoding='utf-8') as f:
                        model_mapping = json.load(f)
                    logger.info(f"Loaded model mapping from {mapping_path}")
                except Exception as e:
                    logger.warning(f"Failed to load model mapping: {e}")

            self.ambiguous_candidates: Dict[str, List[Dict[str, Any]]] = {}
            ambiguous_path = script_dir / "model_mapping_ambiguous.json"
            if ambiguous_path.exists():
                try:
                    with open(ambiguous_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        self.ambiguous_candidates = data
                        logger.info(f"Loaded ambiguous mapping from {ambiguous_path}")
                except Exception as e:
                    logger.warning(f"Failed to load ambiguous mapping: {e}")

            self.synthetic_lookup: Dict[str, Dict[str, Any]] = {}
            self.source_usage: Dict[str, Dict[str, bool]] = {}
            self.last_run_source_stats: Dict[str, Any] = {}
            self.moe_include_patterns, self.moe_exclude_patterns = _load_moe_override_patterns()

            self.aa_client = ArtificialAnalysisClient(
                api_key=validated_config.artificial_analysis_key,
                rate_limit=validated_config.rate_limit_aa,
                max_retries=validated_config.max_retries,
                backoff_factor=validated_config.retry_backoff_factor,
                timeout=validated_config.timeout,
                model_mapping=model_mapping,
                cache_ttl=validated_config.cache_ttl,
                cache_dir=str(script_dir / ".cache")
            )
            self.hf_client = HuggingFaceClient(
                api_key=validated_config.huggingface_key,
                rate_limit=validated_config.rate_limit_hf,
                max_retries=validated_config.max_retries,
                backoff_factor=validated_config.retry_backoff_factor,
                timeout=validated_config.timeout
            )
            ugi_aliases: Dict[str, Union[str, List[str]]] = {}
            ugi_mapping_path = script_dir / "ugi_model_mapping.json"
            if ugi_mapping_path.exists():
                try:
                    with open(ugi_mapping_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        ugi_aliases = data
                        logger.info(f"Loaded UGI model mapping from {ugi_mapping_path}")
                except Exception as e:
                    logger.warning(f"Failed to load UGI model mapping: {e}")

            self.ugi_client = UGILeaderboardClient(
                cache_ttl=validated_config.cache_ttl,
                cache_dir=str(script_dir / ".cache"),
                aliases=ugi_aliases
            )

            open_vlm_aliases: Dict[str, Union[str, List[str]]] = {}
            open_vlm_mapping_path = script_dir / "open_vlm_model_mapping.json"
            if open_vlm_mapping_path.exists():
                try:
                    with open(open_vlm_mapping_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, dict):
                        open_vlm_aliases = data
                        logger.info(f"Loaded Open VLM model mapping from {open_vlm_mapping_path}")
                except Exception as e:
                    logger.warning(f"Failed to load Open VLM model mapping: {e}")

            self.open_vlm_client = OpenVLMLeaderboardClient(
                cache_ttl=validated_config.cache_ttl,
                cache_dir=str(script_dir / ".cache"),
                aliases=open_vlm_aliases
            )

            logger.info("Pipeline initialized successfully")
            logger.info(f"AA API rate limit: {validated_config.rate_limit_aa} req/s")
            logger.info(f"HF API rate limit: {validated_config.rate_limit_hf} req/s")
            logger.info(f"Max retries: {validated_config.max_retries}")

        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise

    @staticmethod
    def _normalize_architecture(raw_value: Any) -> str:
        """Map provider labels to architecture class (dense/moe)."""
        if raw_value is None:
            return "dense"

        if isinstance(raw_value, str):
            lowered = raw_value.strip().lower()
            if any(token in lowered for token in ["moe", "mixture-of-experts", "mixtral", "gemini 1.5"]):
                return "moe"
            if lowered in {"dense", "moe"}:
                return lowered
        return "dense"

    @staticmethod
    def _infer_architecture_from_name(model_name: Optional[str], include_patterns: Optional[List[Pattern[str]]] = None,
                                      exclude_patterns: Optional[List[Pattern[str]]] = None) -> Optional[str]:
        if not model_name:
            return None

        lowered = model_name.strip().lower()

        include_patterns = include_patterns or []
        exclude_patterns = exclude_patterns or []

        if not any(pattern.search(lowered) for pattern in exclude_patterns):
            if any(pattern.search(lowered) for pattern in include_patterns):
                return "moe"

        if any(token in lowered for token in ["moe", "mixture-of-experts", "mixtral"]):
            return "moe"

        for pattern in MOE_NAME_REGEXES:
            if pattern.search(lowered):
                return "moe"

        return None

    @staticmethod
    def _normalize_percentage_value(value: Union[int, float]) -> float:
        if value is None:
            return value
        if isinstance(value, (int, float)) and 0 <= value <= 1:
            return round(value * 100, 3)
        return value

    def _normalize_percentage_sections(self, template: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Ensure all benchmark values adhere to 0-100 percentage scale."""

        for section in ('entity_benchmarks', 'dev_benchmarks'):
            current_section = result.get(section, {}) or {}

            for key, value in current_section.items():
                if isinstance(value, (int, float)):
                    current_section[key] = self._normalize_percentage_value(value)

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

    async def fill_model_data(self,
                       template: Dict,
                       model_info: ModelInfo,
                       session: Optional[aiohttp.ClientSession] = None) -> Dict:
        """Fill template with data from multiple sources"""

        # Validate model info
        validated_model = ModelInfo(**model_info.model_dump() if hasattr(model_info, 'model_dump') else model_info.__dict__)

        logger.info(f"Processing model: {validated_model.name}")

        template_original = copy.deepcopy(template)
        result = copy.deepcopy(template)

        # 1. Fetch from Artificial Analysis
        logger.info("Fetching from Artificial Analysis...")
        aa_data = await self.aa_client.get_model_info(validated_model.name, session=session)
        source_flags = {'aa': False, 'hf': False, 'ugi': False, 'open_vlm': False}

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
            normalized_specs = {}
            for key, value in asdict(specs).items():
                if value is None:
                    continue
                if key == 'architecture':
                    normalized_specs[key] = self._normalize_architecture(value)
                else:
                    normalized_specs[key] = value
            result['model_specs'].update(normalized_specs)

            logger.info(f"  ✓ Retrieved {len(benchmarks)} benchmarks from AA")
            source_flags['aa'] = True
        else:
            logger.warning("  ✗ No data from Artificial Analysis")

        # 2. Fetch from Hugging Face (if model ID provided)
        if validated_model.hf_id:
            logger.info("Fetching from Hugging Face...")
            hf_data = await self.hf_client.get_model_card(validated_model.hf_id, session=session)

            if hf_data:
                hf_info = self.hf_client.extract_info(hf_data)

                # Fill in missing data
                for key, value in hf_info.items():
                    if key in result['dev_benchmarks'] and result['dev_benchmarks'][key] is None:
                        result['dev_benchmarks'][key] = value
                    elif key == 'param_count' and result['model_specs']['param_count'] is None:
                        result['model_specs']['param_count'] = value
                    elif key == 'architecture' and result['model_specs'].get('architecture') is None:
                        result['model_specs']['architecture'] = self._normalize_architecture(value)

                logger.info("  ✓ Retrieved data from Hugging Face")
                source_flags['hf'] = True
            else:
                logger.warning("  ✗ No data from Hugging Face")

        # 3. Fetch UGI leaderboard score
        try:
            ugi_score = await self.ugi_client.get_score(validated_model.name, session=session)
            if ugi_score is not None:
                if result['entity_benchmarks'].get('UGI Leaderboard') is None:
                    result['entity_benchmarks']['UGI Leaderboard'] = ugi_score
                    logger.info("  ✓ Populated UGI Leaderboard score: %s", ugi_score)
                source_flags['ugi'] = True
            else:
                logger.info("  ✗ No UGI leaderboard score for %s", validated_model.name)
        except Exception as exc:
            logger.warning("Failed to fetch UGI leaderboard score: %s", exc)

        # 4. Fetch Open VLM leaderboard score
        try:
            open_vlm_score = await self.open_vlm_client.get_score(validated_model.name, session=session)
            if open_vlm_score is not None and result['entity_benchmarks'].get('Open VLM') is None:
                result['entity_benchmarks']['Open VLM'] = open_vlm_score
                source_flags['open_vlm'] = True
                logger.info("  ✓ Populated Open VLM score: %s", open_vlm_score)
            elif open_vlm_score is None:
                logger.info("  ✗ No Open VLM score for %s", validated_model.name)
            else:
                source_flags['open_vlm'] = True

            extra_open_vlm = await self.open_vlm_client.get_benchmark_scores(validated_model.name, session=session)
            if extra_open_vlm:
                for bench_name, score in extra_open_vlm.items():
                    current_value = result['dev_benchmarks'].get(bench_name)
                    if current_value is None:
                        result['dev_benchmarks'][bench_name] = score
                        logger.info("  ✓ Open VLM provided %s score: %s", bench_name, score)
                source_flags['open_vlm'] = True
        except Exception as exc:
            logger.warning("Failed to fetch Open VLM score: %s", exc)

        # 5. Calculate coverage (only count fields that were originally None and got filled)
        def count_filled_fields(original: Dict, current: Dict) -> int:
            """Count fields that were originally None/empty and now have values"""
            filled = 0
            for key in original.keys():
                if original[key] is None and current.get(key) is not None:
                    filled += 1
            return filled

        self._normalize_percentage_sections(template_original, result)

        filled_entity = count_filled_fields(template_original['entity_benchmarks'], result['entity_benchmarks'])
        filled_dev = count_filled_fields(template_original['dev_benchmarks'], result['dev_benchmarks'])
        filled_community = count_filled_fields(template_original['community_score'], result['community_score'])
        filled_specs = count_filled_fields(template_original['model_specs'], result['model_specs'])

        total_fillable_fields = (
            sum(1 for v in template_original['entity_benchmarks'].values() if v is None) +
            sum(1 for v in template_original['dev_benchmarks'].values() if v is None) +
            sum(1 for v in template_original['community_score'].values() if v is None) +
            sum(1 for v in template_original['model_specs'].values() if v is None)
        )

        total_filled_fields = filled_entity + filled_dev + filled_community + filled_specs

        coverage = (total_filled_fields / total_fillable_fields * 100) if total_fillable_fields > 0 else 100.0
        logger.info(f"  Coverage: {coverage:.1f}% ({total_filled_fields}/{total_fillable_fields} fillable fields)")

        inferred_arch = self._infer_architecture_from_name(
            validated_model.name,
            include_patterns=self.moe_include_patterns,
            exclude_patterns=self.moe_exclude_patterns,
        )
        current_arch = result['model_specs'].get('architecture')
        if inferred_arch == 'moe' and current_arch != 'moe':
            result['model_specs']['architecture'] = 'moe'
        elif not current_arch:
            result['model_specs']['architecture'] = self._normalize_architecture(None)

        self.source_usage[normalize_name(validated_model.name)] = source_flags

        return result

    def _expand_models_with_ambiguous(
        self, models: List[Union[Dict, ModelInfo]]
    ) -> Tuple[List[Union[Dict, ModelInfo]], Dict[str, Dict[str, Any]]]:
        """Append ambiguous AA candidates so they are processed like regular models."""

        if not self.ambiguous_candidates:
            return list(models), {}

        expanded_models = list(models)
        existing_norms = set()
        for entry in expanded_models:
            name = entry.get('name') if isinstance(entry, dict) else getattr(entry, 'name', None)
            if name:
                existing_norms.add(normalize_name(name))

        synthetic_lookup: Dict[str, Dict[str, Any]] = {}
        synthetic_models: List[Dict[str, str]] = []

        for local_name, candidates in self.ambiguous_candidates.items():
            norm_local = normalize_name(local_name)
            local_present = norm_local in existing_norms
            if not local_present:
                logger.debug(
                    "Injecting ambiguous candidates for '%s' even though it was not part of the input batch",
                    local_name,
                )

            for candidate in candidates:
                candidate_name = (
                    candidate.get('name')
                    or candidate.get('slug')
                    or candidate.get('id')
                )
                if not candidate_name:
                    continue

                norm_candidate = normalize_name(candidate_name)
                if norm_candidate in existing_norms:
                    continue

                synthetic_models.append({'name': candidate_name})
                synthetic_lookup[norm_candidate] = {
                    'source_local_name': local_name,
                    'candidate_name': candidate_name,
                    'candidate_id': candidate.get('id'),
                    'slug': candidate.get('slug'),
                }
                existing_norms.add(norm_candidate)

        if synthetic_models:
            logger.info(
                "Added %d ambiguous candidate(s) to processing queue",
                len(synthetic_models),
            )

        return expanded_models + synthetic_models, synthetic_lookup

    def _write_ambiguous_summary(self, results: List[Dict]) -> None:
        """Persist a summary for synthetic ambiguous candidates."""

        if not self.synthetic_lookup:
            return

        ambiguous_results: List[Dict[str, Any]] = []
        for result in results:
            model_name = result.get('model')
            if not model_name:
                continue
            norm_name = normalize_name(model_name)
            origin = self.synthetic_lookup.get(norm_name)
            if not origin:
                continue

            entry = {
                **origin,
                'status': result.get('status'),
            }
            if 'output' in result:
                entry['output'] = result['output']
            if 'error' in result:
                entry['error'] = result['error']
            ambiguous_results.append(entry)

        if not ambiguous_results:
            return

        summary_path = Path(self.config.output_dir) / "ambiguous_candidates_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(ambiguous_results, f, indent=2)
        logger.info("Ambiguous candidate summary saved to %s", summary_path)

        # Reset lookup after writing summary to avoid stale data
        self.synthetic_lookup = {}

    def _resolve_ambiguous_candidates(self, local_name: str) -> List[Dict[str, Any]]:
        candidates = self.ambiguous_candidates.get(local_name)
        if candidates:
            return candidates

        norm_local = normalize_name(local_name)
        for stored_name, stored_candidates in self.ambiguous_candidates.items():
            if normalize_name(stored_name) == norm_local:
                return stored_candidates
        return []

    async def generate_ambiguous_outputs(
        self,
        local_model_name: str,
        template: Dict,
        output_dir: Union[str, Path],
        session: Optional[aiohttp.ClientSession] = None,
        source_file: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Create filled outputs for each ambiguous candidate of a local model."""

        candidates = self._resolve_ambiguous_candidates(local_model_name)
        if not candidates:
            return []

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        base_norm = normalize_name(local_model_name)

        for candidate in candidates:
            candidate_name = (
                candidate.get('name')
                or candidate.get('slug')
                or candidate.get('id')
            )
            if not candidate_name:
                continue

            if normalize_name(candidate_name) == base_norm:
                # Already produced by the main model entry
                continue

            try:
                candidate_info = ModelInfo(name=candidate_name)
                filled_data = await self.fill_model_data(template=template, model_info=candidate_info, session=session)
                safe_name = candidate_name.replace('/', '_').replace('\\', '_')
                output_path = output_dir_path / f"{safe_name}_filled.json"
                self.save_result(filled_data, str(output_path))

                results.append({
                    'file': str(source_file) if source_file else None,
                    'model': candidate_name,
                    'status': 'success',
                    'output': str(output_path),
                    'source_local_name': local_model_name,
                })
            except ValidationError as exc:
                logger.error("Ambiguous candidate validation failed for %s: %s", candidate_name, exc)
                results.append({
                    'file': str(source_file) if source_file else None,
                    'model': candidate_name,
                    'status': 'validation_error',
                    'error': str(exc),
                    'source_local_name': local_model_name,
                })
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Failed to generate ambiguous output for %s: %s", candidate_name, exc)
                results.append({
                    'file': str(source_file) if source_file else None,
                    'model': candidate_name,
                    'status': 'error',
                    'error': str(exc),
                    'source_local_name': local_model_name,
                })

        if results:
            logger.info(
                "Generated %d ambiguous output(s) for %s",
                sum(1 for r in results if r['status'] == 'success'),
                local_model_name,
            )

        return results

    async def _process_single_model(self, model_info: Union[Dict, ModelInfo], template: Dict, synthetic_lookup: Dict, session: aiohttp.ClientSession) -> Dict:
        """Process a single model (helper for batch processing)"""
        try:
            # Validate model info
            if isinstance(model_info, dict):
                validated_model = ModelInfo(**model_info)
            else:
                validated_model = model_info

            # logger.info(f"Processing {validated_model.name}") # Reduced logging for concurrency

            filled_data = await self.fill_model_data(template=template, model_info=validated_model, session=session)

            # Create safe filename
            safe_name = validated_model.name.replace('/', '_').replace('\\', '_')
            output_path = Path(self.config.output_dir) / f"{safe_name}_filled.json"
            self.save_result(filled_data, str(output_path))

            norm_name = normalize_name(validated_model.name)
            origin = synthetic_lookup.get(norm_name)

            result_entry = {
                'model': validated_model.name,
                'status': 'success',
                'output': str(output_path),
                'sources': self.source_usage.get(norm_name, {})
            }
            if origin:
                result_entry['source_local_name'] = origin['source_local_name']

            return result_entry

        except ValidationError as e:
            error_msg = f"Model validation failed: {e}"
            model_label = (
                model_info.get('name', 'unknown') if isinstance(model_info, dict)
                else getattr(model_info, 'name', str(model_info))
            )
            logger.error(f"Error processing {model_label}: {error_msg}")
            norm_name = normalize_name(model_label)
            origin = synthetic_lookup.get(norm_name)
            result_entry = {
                'model': model_label,
                'status': 'validation_error',
                'error': error_msg,
                'sources': self.source_usage.get(norm_name, {})
            }
            if origin:
                result_entry['source_local_name'] = origin['source_local_name']
            return result_entry
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            model_label = (
                model_info.get('name', 'unknown') if isinstance(model_info, dict)
                else getattr(model_info, 'name', str(model_info))
            )
            logger.error(f"Error processing {model_label}: {error_msg}")
            norm_name = normalize_name(model_label)
            origin = synthetic_lookup.get(norm_name)
            result_entry = {
                'model': model_label,
                'status': 'error',
                'error': error_msg,
                'sources': self.source_usage.get(norm_name, {})
            }
            if origin:
                result_entry['source_local_name'] = origin['source_local_name']
            return result_entry

    async def process_batch(self, models: List[Union[Dict, ModelInfo]]) -> List[Dict]:
        """Process multiple models in batch concurrently"""

        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        models_to_process, synthetic_lookup = self._expand_models_with_ambiguous(models)
        self.synthetic_lookup = synthetic_lookup

        # Load and validate template
        template = self.load_template(self.config.template_path)

        logger.info(f"Starting batch processing for {len(models_to_process)} models...")

        async with aiohttp.ClientSession() as session:
            tasks = [
                self._process_single_model(model_info, template, synthetic_lookup, session)
                for model_info in models_to_process
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Handle continue_on_error logic if needed, but gather collects all results.
            # If we want to stop on first error, we should have raised exception.
            # Current logic inside _process_single_model catches exceptions and returns error status.
            # So we just need to filter if continue_on_error is False
            
            if not self.config.continue_on_error:
                # Check if any failed
                first_error = next((r for r in results if r['status'] != 'success'), None)
                if first_error:
                     # In a real async flow, 'stopping' after gather implies we already ran everything.
                     # To support stop-on-error efficiently, we'd need as_completed or wait with FIRST_EXCEPTION.
                     # But since we swallow exceptions in _process_single_model, gather always completes.
                     # For now, we'll just return the results as is, but log.
                     pass

        successful_models = sum(1 for r in results if r['status'] == 'success')
        source_summary = {key: 0 for key in ('aa', 'hf', 'ugi', 'open_vlm')}
        for result in results:
            if result['status'] != 'success':
                continue
            sources = result.get('sources', {}) or {}
            for key in source_summary:
                if sources.get(key):
                    source_summary[key] += 1
        self.last_run_source_stats = {
            'total_success': successful_models,
            'total_models': len(models_to_process),
            'source_counts': source_summary,
        }

        # Save batch summary
        summary_path = Path(self.config.output_dir) / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\nBatch processing complete. {successful_models}/{len(models_to_process)} models successful.")
        logger.info(
            "Data sources used -> AA: %d, HF: %d, UGI: %d, OpenVLM: %d",
            source_summary['aa'],
            source_summary['hf'],
            source_summary['ugi'],
            source_summary['open_vlm'],
        )
        logger.info(f"Summary saved to {summary_path}")

        self._write_ambiguous_summary(results)
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

    # Map models command
    subparsers.add_parser('map-models',
                         help='Generate model mapping file by matching local models with API models')

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


async def launch_interactive_async():
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
        input_folder = input("Input folder containing model JSONs (press Enter for default: ./Models): ").strip()
        input_folder = input_folder if input_folder else "./Models"

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
    output_dir = input("Output folder for results (press Enter for default: ./filled_models): ").strip()
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

    async with aiohttp.ClientSession() as session:
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
                
                filled_data = await pipeline.fill_model_data(template, model_info_obj, session=session)

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

                ambiguous_results = await pipeline.generate_ambiguous_outputs(
                    local_model_name=model_name,
                    template=template,
                    output_dir=output_dir,
                    session=session,
                    source_file=file_path,
                )
                if ambiguous_results:
                    results.extend(ambiguous_results)
                    success_count = sum(1 for entry in ambiguous_results if entry['status'] == 'success')
                    successful += success_count
                    if verbose:
                        for entry in ambiguous_results:
                            status_symbol = '✅' if entry['status'] == 'success' else '⚠️'
                            print(f"   {status_symbol} Ambiguous {entry['model']} -> {entry.get('output') or entry.get('error')}")

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


async def generate_mapping_interactive_async():
    """Interactive tool to generate model mapping file"""
    print("🗺️  Model Mapping Generator")
    print("=" * 50)
    
    # 1. Setup API Client
    aa_key = os.getenv('ARTIFICIAL_ANALYSIS_API_KEY')
    if not aa_key:
        aa_key = input("Artificial Analysis API key: ").strip()
        if not aa_key:
            print("❌ API key is required")
            return

    client = ArtificialAnalysisClient(api_key=aa_key)
    
    # 2. Get Input Folder
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
        print("❌ No JSON files found")
        return

    # 3. Fetch API Models
    print("\n📡 Fetching model list from Artificial Analysis...")
    
    async with aiohttp.ClientSession() as session:
        try:
            api_models = await client.list_models(session=session)
        except Exception as exc:  # pragma: no cover - network errors
            print(f"❌ Failed to fetch models from API: {exc}")
            return
        
        if not api_models:
            print("❌ API returned no models")
            return
        print(f"✅ Retrieved {len(api_models)} models from API")

        # 4. Perform Matching
        print("\n🔄 Matching models...")
        matches: Dict[str, str] = {}
        failures: List[str] = []
        ambiguous_details: Dict[str, List[Dict[str, Any]]] = {}
        
        for file_path in json_files:
            local_names = detect_models_in_file(file_path)
            local_name = local_names[0] if local_names else file_path.stem
            normalized_local = normalize_name(local_name)
            
            found = False
            # Try exact/normalized match first
            for model in api_models:
                api_name = model.get('name', '')
                api_id = model.get('id', '')
                norm_api_name = normalize_name(api_name)
                norm_api_id = normalize_name(api_id)
                
                if normalized_local == norm_api_name or normalized_local == norm_api_id:
                    matches[local_name] = api_id
                    print(f"  ✅ {local_name} -> {api_name} ({api_id}) [Exact/Norm]")
                    found = True
                    break
            
            if not found:
                 # Try substring match
                 candidates = []
                 for model in api_models:
                    api_name = model.get('name', '')
                    api_id = model.get('id', '')
                    norm_api_name = normalize_name(api_name)
                     
                    if normalized_local in norm_api_name or norm_api_name in normalized_local:
                        # Filter out too short matches to avoid false positives
                        if len(normalized_local) > 4 and len(norm_api_name) > 4:
                            candidates.append(model)
                
                 if len(candidates) == 1:
                     candidate = candidates[0]
                     api_name = candidate.get('name', '')
                     matches[local_name] = api_name or candidate.get('slug', '')
                     print(f"  ⚠️ {local_name} -> {api_name} [Partial, mapped to official name]")
                     found = True
                 elif len(candidates) > 1:
                     names_list = ", ".join(model.get('name', 'unknown') for model in candidates)
                     print(f"  ❓ {local_name} -> Multiple candidates: {names_list}")
                     failures.append(local_name)
                     enriched: List[Dict[str, Any]] = []
                     for candidate in candidates:
                         summary = {
                             "name": candidate.get('name'),
                             "slug": candidate.get('slug'),
                             "id": candidate.get('id'),
                             "release_date": candidate.get('release_date'),
                             "benchmarks": client.extract_benchmarks(candidate),
                             "specs": asdict(client.extract_specs(candidate)),
                         }
                         enriched.append(summary)
                         print(
                             f"     ↳ {candidate.get('name')} (slug: {candidate.get('slug')}, id: {candidate.get('id')})"
                         )
                     ambiguous_details[local_name] = enriched
                 else:
                     print(f"  ❌ {local_name} -> No match found")
                     failures.append(local_name)

        # 5. Generate Output
        print("\n" + "=" * 50)
        print(f"Matched: {len(matches)} | Unmatched: {len(failures)}")
        
        if matches:
            output_path = Path("tools/fill-benchmark-pipeline/model_mapping.json")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(matches, f, indent=2)
                
            print(f"\n💾 Generated mapping file at: {output_path}")
            print("Review this file and add any missing mappings manually.")
        
        if ambiguous_details:
            ambiguous_path = Path("tools/fill-benchmark-pipeline/model_mapping_ambiguous.json")
            with open(ambiguous_path, 'w', encoding='utf-8') as f:
                json.dump(ambiguous_details, f, indent=2)
            print(
                f"⚠️  Saved detailed candidate info for ambiguous matches to: {ambiguous_path}"
            )

            per_model_dir = Path("tools/fill-benchmark-pipeline/ambiguous_mappings")
            per_model_dir.mkdir(parents=True, exist_ok=True)

            def _safe_filename(name: str) -> str:
                return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)

            for local_name, candidates in ambiguous_details.items():
                sanitized = _safe_filename(local_name)
                per_model_path = per_model_dir / f"{sanitized}.json"
                payload = {
                    "local_name": local_name,
                    "candidates": candidates,
                }
                with open(per_model_path, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=2)
            print(
                f"🗂️  Also wrote {len(ambiguous_details)} per-model ambiguous files to: {per_model_dir}"
            )
            
            # 6. Optional Verification
            verify = input("\nVerify mappings by fetching info? [y/N]: ").strip().lower()
            if verify == 'y':
                print("\nVerifying...")
                client.model_mapping = matches
                for local, api_id in matches.items():
                    info = await client.get_model_info(local, session=session)
                    status = "✅ OK" if info else "❌ Failed"
                    print(f"  {local} -> {status}")
        else:
            print("\n❌ No matches found to save.")


async def _run_pipeline_cli_async(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    # Handle map-models command
    if args.command == 'map-models':
        try:
            await generate_mapping_interactive_async()
            return 0
        except KeyboardInterrupt:
            print("\n\n❌ Operation cancelled")
            return 1
        except Exception as e:
             logger.error(f"Mapping generation failed: {e}")
             return 1

    # Handle interactive launch command
    if args.command == 'launch':
        try:
            await launch_interactive_async()
            return 0
        except KeyboardInterrupt:
            print("\n\n❌ Operation cancelled by user")
            return 1
        except Exception as e:  # pragma: no cover - defensive logging
            logger.error(f"Interactive mode failed: {e}")
            return 1

    # Set logging level for non-interactive modes
    log_level = 'DEBUG' if args.verbose else args.log_level
    logging.getLogger().setLevel(getattr(logging, log_level))

    try:
        config_data: Dict[str, Any] = {}

        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config_data = load_config_from_file(args.config)

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

        if args.aa_key:
            config_data['artificial_analysis_key'] = args.aa_key
        elif 'ARTIFICIAL_ANALYSIS_API_KEY' in os.environ:
            config_data['artificial_analysis_key'] = os.environ['ARTIFICIAL_ANALYSIS_API_KEY']

        if args.hf_key:
            config_data['huggingface_key'] = args.hf_key
        elif 'HUGGINGFACE_API_KEY' in os.environ:
            config_data['huggingface_key'] = os.environ['HUGGINGFACE_API_KEY']

        if not args.models and 'models' not in config_data:
            parser.error("--models is required (or specify in config file)")
        if not args.template and 'template_path' not in config_data:
            parser.error("--template is required (or specify in config file)")

        if args.models:
            models = load_models(args.models)
        else:
            models = config_data['models']

        pipeline_config = PipelineConfig(**config_data)
        pipeline = LLMBenchmarkPipeline(pipeline_config)
        results = await pipeline.process_batch(models)

        print("\n" + "=" * 60)
        print("BATCH PROCESSING SUMMARY")
        print("=" * 60)

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

        return 0 if error_count == 0 else 1

    except ValidationError as e:
        logger.error(f"Configuration validation failed: {e}")
        return 1
    except Exception as e:  # pragma: no cover - safety net
        logger.error(f"Unexpected error: {e}")
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point with configuration management."""

    parser = create_parser()
    args = parser.parse_args(argv)
    return asyncio.run(_run_pipeline_cli_async(args, parser))


if __name__ == "__main__":
    raise SystemExit(main())
