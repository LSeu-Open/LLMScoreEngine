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
Tests for the scoring configuration.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import pytest
import types
from config import scoring_config

# ------------------------------------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------------------------------------

def test_score_scale_and_bounds():
    """Tests the score scale and bounds constants."""
    print("\n[Testing Score Scale and Bounds]")
    print(f"  - SCORE_SCALE: {scoring_config.SCORE_SCALE} (Type: {type(scoring_config.SCORE_SCALE).__name__})")
    assert isinstance(scoring_config.SCORE_SCALE, int)
    assert scoring_config.SCORE_SCALE == 100
    
    print(f"  - SCORE_BOUNDS: {scoring_config.SCORE_BOUNDS} (Type: {type(scoring_config.SCORE_BOUNDS).__name__})")
    assert isinstance(scoring_config.SCORE_BOUNDS, types.MappingProxyType)
    assert scoring_config.SCORE_BOUNDS == {"min": 0, "max": 100}

def test_score_weights():
    """Tests the score weights."""
    weights = scoring_config.SCORE_WEIGHTS
    print("\n[Testing Score Weights]")
    print(f"  - Weights: {weights}")
    assert isinstance(weights, types.MappingProxyType)
    
    expected_keys = ['entity_benchmarks', 'dev_benchmarks', 'community_score', 'technical_score']
    are_keys_present = all(key in weights for key in expected_keys)
    print(f"  - All expected keys present: {are_keys_present}")
    assert are_keys_present
    
    # Test that weights sum to the score scale
    total_weight = sum(weights.values())
    print(f"  - Sum of weights: {total_weight} (Expected: {scoring_config.SCORE_SCALE})")
    assert total_weight == scoring_config.SCORE_SCALE

    # Test that weights are immutable
    print("  - Testing immutability...")
    with pytest.raises(TypeError):
        weights['new_category'] = 10
    print("  - Immutability test passed.")

def test_benchmark_weights():
    """Tests the benchmark weights configuration."""
    weights = scoring_config.BENCHMARK_WEIGHTS
    print("\n[Testing Benchmark Weights]")
    print(f"  - BENCHMARK_WEIGHTS type: {type(weights).__name__}")
    assert isinstance(weights, types.MappingProxyType)
    
    assert 'entity_benchmarks' in weights
    assert 'dev_benchmarks' in weights
    print("  - 'entity_benchmarks' and 'dev_benchmarks' keys are present.")
    
    print(f"  - 'entity_benchmarks' type: {type(weights['entity_benchmarks']).__name__}")
    assert isinstance(weights['entity_benchmarks'], types.MappingProxyType)
    print(f"  - 'dev_benchmarks' type: {type(weights['dev_benchmarks']).__name__}")
    assert isinstance(weights['dev_benchmarks'], types.MappingProxyType)

    # Test that benchmark weights are immutable
    print("  - Testing immutability...")
    with pytest.raises(TypeError):
        weights['new_benchmark'] = {}
    print("  - Immutability test passed.")

def test_technical_score_params():
    """Tests the technical score parameters."""
    params = scoring_config.TECHNICAL_SCORE_PARAMS
    print("\n[Testing Technical Score Parameters]")
    print(f"  - TECHNICAL_SCORE_PARAMS type: {type(params).__name__}")
    assert isinstance(params, types.MappingProxyType)
    
    expected_keys = ['input_price', 'output_price', 'context_window', 'size_perf_ratio']
    are_keys_present = all(key in params for key in expected_keys)
    print(f"  - All expected keys present: {are_keys_present}")
    assert are_keys_present

    # Test immutability
    print("  - Testing immutability...")
    with pytest.raises(TypeError):
        params['new_param'] = {}
    print("  - Immutability test passed.")

    # Test nested structure for input and output prices
    for price_key in ['input_price', 'output_price']:
        price_params = params[price_key]
        print(f"  - '{price_key}' params type: {type(price_params).__name__}")
        assert isinstance(price_params, types.MappingProxyType)
        price_keys = ['max_points', 'coefficient', 'intercept', 'high_price_cutoff', 'high_price_points']
        are_price_keys_present = all(key in price_params for key in price_keys)
        print(f"    - All expected keys present: {are_price_keys_present}")
        assert are_price_keys_present

    # Test nested structure for context window and size/performance ratio
    context_params = params['context_window']
    print(f"  - 'context_window' params type: {type(context_params).__name__}")
    assert isinstance(context_params, types.MappingProxyType)
    context_keys = ['max_points', 'coefficient', 'intercept', 'log_base', 'low_cw_cutoff', 'low_cw_points']
    assert all(key in context_params for key in context_keys)

    size_params = params['size_perf_ratio']
    print(f"  - 'size_perf_ratio' params type: {type(size_params).__name__}")
    assert isinstance(size_params, types.MappingProxyType)
    size_keys = ['max_points', 'base_points', 'scaling_factor', 'size_tiers', 'default_size_factor']
    assert all(key in size_params for key in size_keys)

def test_model_architecture_factors():
    """Tests the model architecture factors."""
    factors = scoring_config.MODEL_ARCHITECTURE_FACTORS
    print("\n[Testing Model Architecture Factors]")
    print(f"  - MODEL_ARCHITECTURE_FACTORS type: {type(factors).__name__}")
    assert isinstance(factors, types.MappingProxyType)
    
    print(f"  - 'default' factor present: {'default' in factors}, Value: {factors.get('default')}")
    assert 'default' in factors
    assert factors['default'] == 1.0

    # Test immutability
    print("  - Testing immutability...")
    with pytest.raises(TypeError):
        factors['new_arch'] = 1.5
    print("  - Immutability test passed.")

def test_community_score_bounds():
    """Tests the community score bounds."""
    bounds = scoring_config.COMMUNITY_SCORE_BOUNDS
    print("\n[Testing Community Score Bounds]")
    print(f"  - COMMUNITY_SCORE_BOUNDS type: {type(bounds).__name__}")
    assert isinstance(bounds, types.MappingProxyType)
    
    expected_keys = ['lm_sys_arena_score', 'hf_score']
    are_keys_present = all(key in bounds for key in expected_keys)
    print(f"  - All expected keys present: {are_keys_present}")
    assert are_keys_present
    
    # Test immutability
    print("  - Testing immutability...")
    with pytest.raises(TypeError):
        bounds['new_bound'] = {}
    print("  - Immutability test passed.")

def test_hugging_face_score_params():
    """Tests the Hugging Face score parameters."""
    params = scoring_config.HUGGING_FACE_SCORE_PARAMS
    print("\n[Testing Hugging Face Score Parameters]")
    print(f"  - HUGGING_FACE_SCORE_PARAMS type: {type(params).__name__}")
    assert isinstance(params, types.MappingProxyType)
    
    expected_keys = ['downloads', 'likes', 'age_months']
    are_keys_present = all(key in params for key in expected_keys)
    print(f"  - All expected keys present: {are_keys_present}")
    assert are_keys_present

    # Test immutability
    print("  - Testing immutability...")
    with pytest.raises(TypeError):
        params['new_param'] = {}
    print("  - Immutability test passed.")
    
    # Test nested structure for 'age_months'
    age_params = params['age_months']
    print(f"  - 'age_months' params type: {type(age_params).__name__}")
    assert isinstance(age_params, types.MappingProxyType)
    age_keys = [
        'tier1_months', 'tier1_slope', 'tier2_months', 'tier2_base_points', 
        'tier2_slope', 'tier3_months', 'tier3_base_points', 'tier3_slope', 
        'stable_points', 'max_points'
    ]
    are_age_keys_present = all(key in age_params for key in age_keys)
    print(f"  - All expected age_months keys present: {are_age_keys_present}")
    assert are_age_keys_present 