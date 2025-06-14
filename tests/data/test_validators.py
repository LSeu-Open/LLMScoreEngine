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
Tests for the data validation utilities.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import pytest
import copy

from model_scoring.data import validators
from model_scoring.core import exceptions
from model_scoring.core.constants import REQUIRED_SECTIONS
from config import scoring_config as default_scoring_config

# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------

@pytest.fixture
def validator_instance():
    """Provides an instance of ModelDataValidator with the default config."""
    return validators.ModelDataValidator(scoring_config=default_scoring_config)

@pytest.fixture
def valid_model_data():
    """Provides a deep copy of a valid model data structure for testing."""
    data = {
        "entity_benchmarks": {
            "artificial_analysis": 85,
            "OpenCompass": 90
        },
        "dev_benchmarks": {
            "MMLU": 75,
            "GSM-8K": 80
        },
        "model_specs": {
            "architecture": "dense",
            "param_count": 7000000000,
            "context_window": 8192,
            "price_per_million_tokens_input": 1.5,
            "price_per_million_tokens_output": 2.0
        },
        "community_score": {
            "lm_sys_arena_elo_rating": 1250,
            "hf_score": 8.5,
            "hf_downloads_last_30_days": 50000,
            "hf_likes": 1500,
            "model_age_months": 6
        }
    }
    # Mock REQUIRED_SECTIONS based on this valid data for isolated testing
    REQUIRED_SECTIONS['entity_benchmarks'] = ["artificial_analysis", "OpenCompass"]
    REQUIRED_SECTIONS['dev_benchmarks'] = ["MMLU", "GSM-8K"]
    REQUIRED_SECTIONS['model_specs'] = ["architecture", "param_count", "context_window", "price_per_million_tokens_input", "price_per_million_tokens_output"]
    REQUIRED_SECTIONS['community_score'] = ["lm_sys_arena_elo_rating", "hf_score", "hf_downloads_last_30_days", "hf_likes", "model_age_months"]
    
    return copy.deepcopy(data)

# ------------------------------------------------------------------------------------------------
# Tests for ModelDataValidator
# ------------------------------------------------------------------------------------------------

class TestModelDataValidator:

    def test_validate_benchmarks_success(self, valid_model_data, validator_instance):
        """Test benchmark validation with valid data, including score normalization."""
        print("\n--- Testing Benchmark Validation (Success) ---")
        print(f"Original score: {valid_model_data['entity_benchmarks']['artificial_analysis']}")
        validator_instance.validate_benchmarks(valid_model_data, 'entity_benchmarks', 'TestModel')
        print(f"Normalized score: {valid_model_data['entity_benchmarks']['artificial_analysis']}")
        # Check that score was normalized correctly (85 -> 0.85)
        assert valid_model_data['entity_benchmarks']['artificial_analysis'] == 0.85
        print("✅ Benchmark validation and normalization successful.")

    def test_validate_benchmarks_missing_field(self, valid_model_data, validator_instance):
        """Test that BenchmarkScoreError is raised for a missing benchmark."""
        print("\n--- Testing Benchmark Validation (Missing Field) ---")
        field_to_remove = 'OpenCompass'
        del valid_model_data['entity_benchmarks'][field_to_remove]
        print(f"Testing with '{field_to_remove}' benchmark removed...")
        with pytest.raises(exceptions.BenchmarkScoreError, match=f"Missing benchmark '{field_to_remove}'"):
            validator_instance.validate_benchmarks(valid_model_data, 'entity_benchmarks', 'TestModel')
        print(f"✅ Correctly raised BenchmarkScoreError for missing field.")

    def test_validate_benchmarks_invalid_type(self, valid_model_data, validator_instance):
        """Test that BenchmarkScoreError is raised for a non-numeric score."""
        print("\n--- Testing Benchmark Validation (Invalid Type) ---")
        invalid_value = "not-a-number"
        valid_model_data['dev_benchmarks']['MMLU'] = invalid_value
        print(f"Testing with score set to: '{invalid_value}'...")
        with pytest.raises(exceptions.BenchmarkScoreError, match="Invalid score type for 'MMLU'"):
            validator_instance.validate_benchmarks(valid_model_data, 'dev_benchmarks', 'TestModel')
        print("✅ Correctly raised BenchmarkScoreError for invalid type.")

    def test_validate_benchmarks_out_of_bounds(self, valid_model_data, validator_instance):
        """Test that BenchmarkScoreError is raised for a score outside the 0-100 range."""
        print("\n--- Testing Benchmark Validation (Out of Bounds) ---")
        invalid_score = 101
        valid_model_data['dev_benchmarks']['GSM-8K'] = invalid_score
        print(f"Testing with score set to: {invalid_score}...")
        with pytest.raises(exceptions.BenchmarkScoreError, match="must be between 0 and 100"):
            validator_instance.validate_benchmarks(valid_model_data, 'dev_benchmarks', 'TestModel')
        print("✅ Correctly raised BenchmarkScoreError for out-of-bounds score.")
    
    def test_validate_model_specs_success(self, valid_model_data):
        """Test model spec validation with valid data."""
        print("\n--- Testing Model Specs Validation (Success) ---")
        print("Testing with valid model_specs data...")
        # This should pass without raising an exception
        validators.ModelDataValidator.validate_model_specs(valid_model_data['model_specs'], 'TestModel')
        print("✅ Model specs validation successful.")

    def test_validate_model_specs_missing_field(self, valid_model_data):
        """Test that ModelSpecificationError is raised for a missing spec."""
        print("\n--- Testing Model Specs Validation (Missing Field) ---")
        field_to_remove = 'param_count'
        del valid_model_data['model_specs'][field_to_remove]
        print(f"Testing with '{field_to_remove}' spec removed...")
        with pytest.raises(exceptions.ModelSpecificationError, match=f"Missing specification '{field_to_remove}'"):
            validators.ModelDataValidator.validate_model_specs(valid_model_data['model_specs'], 'TestModel')
        print(f"✅ Correctly raised ModelSpecificationError for missing spec.")

    def test_validate_model_specs_invalid_architecture(self, valid_model_data):
        """Test that ModelSpecificationError is raised for an invalid architecture type."""
        print("\n--- Testing Model Specs Validation (Invalid Architecture Type) ---")
        invalid_arch = 123
        valid_model_data['model_specs']['architecture'] = invalid_arch
        print(f"Testing with architecture set to: {invalid_arch}...")
        with pytest.raises(exceptions.ModelSpecificationError, match="Invalid type for 'architecture'"):
            validators.ModelDataValidator.validate_model_specs(valid_model_data['model_specs'], 'TestModel')
        print("✅ Correctly raised ModelSpecificationError for invalid architecture type.")

    def test_validate_model_specs_non_positive_value(self, valid_model_data):
        """Test that ModelSpecificationError is raised for a non-positive numeric spec."""
        print("\n--- Testing Model Specs Validation (Non-Positive Value) ---")
        invalid_value = 0
        valid_model_data['model_specs']['context_window'] = invalid_value
        print(f"Testing with context_window set to: {invalid_value}...")
        with pytest.raises(exceptions.ModelSpecificationError, match="must be positive"):
            validators.ModelDataValidator.validate_model_specs(valid_model_data['model_specs'], 'TestModel')
        print("✅ Correctly raised ModelSpecificationError for non-positive value.")

    def test_validate_community_score_success(self, valid_model_data, validator_instance):
        """Test community score validation with valid data."""
        print("\n--- Testing Community Score Validation (Success) ---")
        print("Testing with valid community_score data...")
        # This should pass without raising an exception
        validator_instance.validate_community_score(valid_model_data['community_score'], 'TestModel')
        print("✅ Community score validation successful.")

    def test_validate_community_score_out_of_bounds(self, valid_model_data, validator_instance):
        """Test that CommunityScoreError is raised for an out-of-bounds hf_score."""
        print("\n--- Testing Community Score Validation (Out of Bounds) ---")
        invalid_score = 11
        valid_model_data['community_score']['hf_score'] = invalid_score
        print(f"Testing with hf_score set to: {invalid_score}...")
        with pytest.raises(exceptions.CommunityScoreError, match="must be between 0 and 10"):
            validator_instance.validate_community_score(valid_model_data['community_score'], 'TestModel')
        print("✅ Correctly raised CommunityScoreError for out-of-bounds score.")
            
    def test_validate_community_score_missing_field(self, valid_model_data, validator_instance):
        """Test that CommunityScoreError is raised for a missing community score field."""
        print("\n--- Testing Community Score Validation (Missing Field) ---")
        field_to_remove = 'lm_sys_arena_elo_rating'
        del valid_model_data['community_score'][field_to_remove]
        print(f"Testing with '{field_to_remove}' field removed...")
        with pytest.raises(exceptions.CommunityScoreError, match=f"Missing community score field '{field_to_remove}'"):
            validator_instance.validate_community_score(valid_model_data['community_score'], 'TestModel')
        print(f"✅ Correctly raised CommunityScoreError for missing field.")

# ------------------------------------------------------------------------------------------------
# Tests for validate_model_data
# ------------------------------------------------------------------------------------------------

def test_validate_model_data_success(valid_model_data):
    """Test the main validation function with a completely valid data object."""
    print("\n--- Testing Full Validation (Success) ---")
    print("Testing with a fully valid model data object...")
    # This should pass without raising an exception
    validators.validate_model_data(valid_model_data, "TestModel")
    print("✅ Full data object validation successful.")

def test_validate_model_data_missing_section(valid_model_data):
    """Test that ModelDataValidationError is raised for a missing top-level section."""
    print("\n--- Testing Full Validation (Missing Section) ---")
    section_to_remove = 'dev_benchmarks'
    del valid_model_data[section_to_remove]
    print(f"Testing with top-level section '{section_to_remove}' removed...")
    with pytest.raises(exceptions.ModelDataValidationError, match=f"Missing required section '{section_to_remove}'"):
        validators.validate_model_data(valid_model_data, "TestModel")
    print(f"✅ Correctly raised ModelDataValidationError for missing section.") 