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
Tests for the models scoring.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import pytest
from model_scoring.scoring.models_scoring import ModelScorer
from config import scoring_config

@pytest.fixture
def scorer():
    """Returns a ModelScorer instance."""
    return ModelScorer(model_name="TestModel")

@pytest.fixture
def entity_benchmark_data():
    """Provides sample entity benchmark scores."""
    return {
        'artificial_analysis': 0.85,
        'OpenCompass': 0.90,
        'non_existent_bench': 1.0, # Should be ignored
        'Dubesord_LLM': None # Should be ignored
    }

@pytest.fixture
def dev_benchmark_data():
    """Provides sample dev benchmark scores."""
    return {
        'MMLU': 0.75,
        'GSM-8K': 0.80,
        'HumanEval': 0.88
    }

# ------------------------------------------------------------------------------------------------
# Benchmark Score Tests
# ------------------------------------------------------------------------------------------------

def test_calculate_entity_benchmarks(scorer, entity_benchmark_data):
    """Tests the calculation of the entity benchmarks score."""
    # Weights for artificial_analysis (10) and OpenCompass (10) are used.
    # Total weight used = 10 + 10 = 20
    # Raw score = (0.85 * 10) + (0.90 * 10) = 8.5 + 9.0 = 17.5
    # Average performance = 17.5 / 20 = 0.875
    # Final score = 0.875 * 30 (category weight) = 26.25
    expected_score = 26.25
    actual_score = scorer.calculate_entity_benchmarks(entity_benchmark_data)
    print(f"\n[Entity Benchmarks] Calculated: {actual_score:.4f}, Expected: {expected_score:.4f}")
    assert actual_score == expected_score
    assert scorer.calculate_entity_benchmarks({}) == 0.0

def test_calculate_dev_benchmarks(scorer, dev_benchmark_data):
    """Tests the calculation of the dev benchmarks score."""
    # Weights: MMLU (3.0), GSM-8K (3.0), HumanEval (1.0)
    # Total weight = 3.0 + 3.0 + 1.0 = 7.0
    # Raw score = (0.75 * 3.0) + (0.80 * 3.0) + (0.88 * 1.0) = 2.25 + 2.4 + 0.88 = 5.53
    # Average performance = 5.53 / 7.0 = 0.79
    # Final score = 0.79 * 30 (category weight) = 23.7
    expected_score = ( ( (0.75 * 3.0) + (0.80 * 3.0) + (0.88 * 1.0) ) / 7.0 ) * 30
    actual_score = scorer.calculate_dev_benchmarks(dev_benchmark_data)
    print(f"\n[Dev Benchmarks] Calculated: {actual_score:.4f}, Expected: {expected_score:.4f}")
    assert round(actual_score, 2) == round(expected_score, 2)
    assert scorer.calculate_dev_benchmarks({}) == 0.0

# ------------------------------------------------------------------------------------------------
# Community Score Tests
# ------------------------------------------------------------------------------------------------

def test_calculate_community_score(scorer):
    """Tests community score calculation with different inputs."""
    category_weight = scoring_config.SCORE_WEIGHTS['community_score']
    print("\n[Community Score]")

    # Both scores provided
    # ELO: 1300 -> (1300 - 1000) / (1500 - 1000) = 0.6. Weighted: 0.6 * (20/2) = 6.0
    # HF: 8.5 -> (8.5 / 10) * (20/2) = 8.5
    # Total = 6.0 + 8.5 = 14.5
    score1 = scorer.calculate_community_score(1300, 8.5)
    print(f"  - ELO: 1300, HF: 8.5 -> Calculated: {score1}, Expected: 14.5")
    assert score1 == 14.5

    # Only ELO provided
    # ELO: 1300 -> 0.6 * 20 = 12.0
    score2 = scorer.calculate_community_score(1300, None)
    print(f"  - ELO: 1300, HF: None -> Calculated: {score2}, Expected: 12.0")
    assert score2 == 12.0

    # Only HF provided
    # HF: 8.5 -> (8.5 / 10) * 20 = 17.0
    score3 = scorer.calculate_community_score(None, 8.5)
    print(f"  - ELO: None, HF: 8.5 -> Calculated: {score3}, Expected: 17.0")
    assert score3 == 17.0

    # None provided
    score4 = scorer.calculate_community_score(None, None)
    print(f"  - ELO: None, HF: None -> Calculated: {score4}, Expected: 0.0")
    assert score4 == 0.0

    # Test clamping
    score5 = scorer.calculate_community_score(2000, 12)
    print(f"  - ELO: 2000, HF: 12 (clamped) -> Calculated: {score5}, Expected: {category_weight}")
    assert score5 == category_weight

# ------------------------------------------------------------------------------------------------
# Technical Score Tests
# ------------------------------------------------------------------------------------------------

def test_calculate_price_component_score(scorer):
    """Test price component scoring for both input and output prices."""
    # Test input price
    params_in = scoring_config.TECHNICAL_SCORE_PARAMS['input_price']
    print("\n[Input Price Score]")
    assert scorer._calculate_price_component_score(None, 'input_price') == 0.0
    assert scorer._calculate_price_component_score(0, 'input_price') == params_in['max_points']
    assert scorer._calculate_price_component_score(params_in['high_price_cutoff'], 'input_price') == params_in['high_price_points']
    
    # Test output price
    params_out = scoring_config.TECHNICAL_SCORE_PARAMS['output_price']
    print("\n[Output Price Score]")
    assert scorer._calculate_price_component_score(None, 'output_price') == 0.0
    assert scorer._calculate_price_component_score(0, 'output_price') == params_out['max_points']
    assert scorer._calculate_price_component_score(params_out['high_price_cutoff'], 'output_price') == params_out['high_price_points']

def test_calculate_context_score(scorer):
    params = scoring_config.TECHNICAL_SCORE_PARAMS['context_window']
    print("\n[Context Score]")
    
    score1 = scorer._calculate_context_score(None)
    print(f"  - Context: None -> Score: {score1}")
    assert score1 == 0.0

    score2 = scorer._calculate_context_score(params['low_cw_cutoff'] - 1)
    print(f"  - Context: {params['low_cw_cutoff'] - 1} (below cutoff) -> Score: {score2}")
    assert score2 == params['low_cw_points']

    # A mid-range context
    context = 32768
    # expected = 0.571 * log2(32768) + -5.929 = 0.571 * 15 - 5.929 = 8.565 - 5.929 = 2.636
    score3 = scorer._calculate_context_score(context)
    print(f"  - Context: {context} -> Score: {score3:.4f}")
    assert round(score3, 3) == 2.636

def test_calculate_size_perf_ratio(scorer):
    print("\n[Size/Perf Ratio]")
    # Test with a high-performing large model
    # Benchmark 85, 70B params, dense arch
    # base_size_factor = 0.80, arch_factor = 1.0 -> total_eff = 0.80
    # combined_score = (85/100) * 0.80 = 0.68
    # points = 1.0 + (5.0 * 0.68) = 1.0 + 3.4 = 4.4
    score1 = scorer.calculate_size_perf_ratio(85.0, 70_000_000_000, 'dense')
    print(f"  - Large model (85 benchmark, 70B, dense) -> Score: {score1:.4f}")
    assert round(score1, 1) == 4.4

    # Test with a small, efficient model (MoE)
    # Benchmark 70, 2.8B params, moe arch
    # base_size_factor = 1.00, arch_factor = 1.2 -> total_eff = 1.2
    # combined_score = (70/100) * 1.2 = 0.84
    # points = 1.0 + (5.0 * 0.84) = 1.0 + 4.2 = 5.2
    score2 = scorer.calculate_size_perf_ratio(70.0, 2_800_000_000, 'moe')
    print(f"  - Small model (70 benchmark, 2.8B, moe) -> Score: {score2:.4f}")
    assert round(score2, 1) == 5.2

def test_calculate_technical_score(scorer):
    # input_price(1.5) = 4.0 - 0.15*1.5 = 3.775
    # output_price(3.0) = 4.0 - 0.0375*3.0 = 3.8875
    # context_score(131072) = 3.778
    # ratio(88, 7B, dense) = 5.18
    # total = 3.775 + 3.8875 + 3.778 + 5.18 = 16.6205
    expected_total = 16.62
    
    in_price, out_price, ctx_val, bench_val, params_val, arch_val = 1.5, 3.0, 131072, 88.0, 7_000_000_000, 'dense'
    
    input_price_score = scorer._calculate_price_component_score(in_price, 'input_price')
    output_price_score = scorer._calculate_price_component_score(out_price, 'output_price')
    context_score = scorer._calculate_context_score(ctx_val)
    ratio_score = scorer.calculate_size_perf_ratio(bench_val, params_val, arch_val)
    actual_total = scorer.calculate_technical_score(in_price, out_price, ctx_val, bench_val, params_val, arch_val)
    
    print("\n[Technical Score Breakdown]")
    print(f"  - Input Price Score:   {input_price_score:.4f}")
    print(f"  - Output Price Score:  {output_price_score:.4f}")
    print(f"  - Context Score: {context_score:.4f}")
    print(f"  - Ratio Score:   {ratio_score:.4f}")
    print("  ---")
    print(f"  - Actual Total:   {actual_total}")
    print(f"  - Expected Total: {expected_total}")

    assert actual_total == expected_total

# This requires the final method, which was not fully visible
# I'll create a placeholder for now.
# def test_calculate_final_score(scorer):
#     pass 

def test_calculate_final_score(scorer, entity_benchmark_data, dev_benchmark_data):
    """Tests the final score calculation by integrating all components."""
    # This is a full integration test, so we use the real methods.
    # We will instantiate a new scorer to avoid side-effects from other tests.
    integration_scorer = ModelScorer(model_name="IntegrationTestModel")

    # 1. Define mock inputs for all categories
    community_inputs = {'lm_sys_arena_elo_rating': 1300, 'hf_score': 8.5}
    tech_inputs = {
        'input_price': 1.5,
        'output_price': 3.0,
        'context_window': 131072,
        'param_count': 7_000_000_000,
        'architecture': 'dense'
    }

    # 2. Calculate expected scores for each category
    entity_score = integration_scorer.calculate_entity_benchmarks(entity_benchmark_data)
    dev_score = integration_scorer.calculate_dev_benchmarks(dev_benchmark_data)
    community_score = integration_scorer.calculate_community_score(**community_inputs)
    
    # The technical score's ratio component needs a benchmark score. 
    # The actual implementation uses a combined raw benchmark performance.
    # Let's use a hypothetical combined benchmark score for this test.
    # This is a known simplification for this test.
    all_benchmark_weights = {
        **integration_scorer.config.BENCHMARK_WEIGHTS['entity_benchmarks'],
        **integration_scorer.config.BENCHMARK_WEIGHTS['dev_benchmarks']
    }
    all_benchmark_scores = {**entity_benchmark_data, **dev_benchmark_data}
    
    score = 0.0
    total_weight = 0.0
    for bench_key, result in all_benchmark_scores.items():
        if bench_key in all_benchmark_weights and result is not None:
            score += (result * all_benchmark_weights[bench_key])
            total_weight += all_benchmark_weights[bench_key]
    
    overall_benchmark_score = (score / total_weight * 100) if total_weight > 0 else 0.0
    tech_inputs['benchmark_score'] = overall_benchmark_score
    technical_score = integration_scorer.calculate_technical_score(**tech_inputs)
    
    # 3. Sum of component scores for the final expectation
    expected_final_score = entity_score + dev_score + community_score + technical_score
    
    # 4. Call the actual final_score method with the same inputs
    # This requires the actual implementation of calculate_final_score to be correct.
    final_score = integration_scorer.calculate_final_score(
        entity_benchmarks=entity_benchmark_data,
        dev_benchmarks=dev_benchmark_data,
        community_inputs=community_inputs,
        tech_inputs=tech_inputs
    )
    
    print("\n[Final Score Breakdown]")
    print(f"  - Entity Score:    {entity_score:.2f}")
    print(f"  - Dev Score:       {dev_score:.2f}")
    print(f"  - Community Score: {community_score:.2f}")
    print(f"  - Technical Score: {technical_score:.2f}")
    print("  ---")
    print(f"  - Calculated Final Score: {final_score}")
    print(f"  - Expected Final Score:   {round(expected_final_score, 4)}")
    
    assert final_score == round(expected_final_score, 4)

# ------------------------------------------------------------------------------------------------
# Edge Case Tests
# ------------------------------------------------------------------------------------------------

def test_calculate_benchmarks_edge_cases(scorer):
    """Tests benchmark calculations with edge case data."""
    print("\n[Benchmark Edge Cases]")
    # Test with all zero scores, should result in zero.
    entity_data_zeros = {'artificial_analysis': 0.0, 'OpenCompass': 0.0}
    score1 = scorer.calculate_entity_benchmarks(entity_data_zeros)
    print(f"  - All zero scores -> Score: {score1}")
    assert score1 == 0.0

    # Test with benchmarks that are not in the config, resulting in zero weight.
    dev_data_no_weight = {'non_existent_bench': 0.9, 'another_one': 0.5}
    score2 = scorer.calculate_dev_benchmarks(dev_data_no_weight)
    print(f"  - No valid benchmarks -> Score: {score2}")
    assert score2 == 0.0

    # Test with a valid benchmark that has a score of None.
    dev_data_none_score = {'MMLU': None}
    score3 = scorer.calculate_dev_benchmarks(dev_data_none_score)
    print(f"  - Benchmark with None score -> Score: {score3}")
    assert score3 == 0.0

def test_calculate_community_score_bounds(scorer):
    """Tests community score calculation at the exact bounds."""
    elo_bounds = scoring_config.COMMUNITY_SCORE_BOUNDS['lm_sys_arena_score']
    hf_bounds = scoring_config.COMMUNITY_SCORE_BOUNDS['hf_score']
    category_weight = scoring_config.SCORE_WEIGHTS['community_score']
    print("\n[Community Score Bounds]")

    # Test at lower bound (should result in 0 for that part).
    score1 = scorer.calculate_community_score(elo_bounds['min'], hf_bounds['min'])
    print(f"  - Lower bounds ({elo_bounds['min']}, {hf_bounds['min']}) -> Score: {score1}")
    assert score1 == 0.0

    # Test at upper bound (should result in max points for each part).
    score2 = scorer.calculate_community_score(elo_bounds['max'], hf_bounds['max'])
    print(f"  - Upper bounds ({elo_bounds['max']}, {hf_bounds['max']}) -> Score: {score2}")
    assert score2 == category_weight

def test_technical_score_edge_cases():
    """Tests technical score components with edge case values."""
    # Use a dedicated scorer to prevent side-effects from other tests.
    edge_scorer = ModelScorer(model_name="EdgeCaseScorer")
    
    # Price edge cases
    input_price_params = scoring_config.TECHNICAL_SCORE_PARAMS['input_price']
    output_price_params = scoring_config.TECHNICAL_SCORE_PARAMS['output_price']
    
    assert edge_scorer._calculate_price_component_score(-1, 'input_price') == input_price_params['max_points']
    assert edge_scorer._calculate_price_component_score(999, 'input_price') == input_price_params['high_price_points']
    assert edge_scorer._calculate_price_component_score(-1, 'output_price') == output_price_params['max_points']
    assert edge_scorer._calculate_price_component_score(999, 'output_price') == output_price_params['high_price_points']

    # Context window edge cases
    context_params = scoring_config.TECHNICAL_SCORE_PARAMS['context_window']
    assert edge_scorer._calculate_context_score(0) == context_params['low_cw_points']
    assert edge_scorer._calculate_context_score(9999999) == context_params['max_points']

    # Size/performance ratio edge cases
    size_params = scoring_config.TECHNICAL_SCORE_PARAMS['size_perf_ratio']
    # Very poor benchmark score should result in base points
    assert edge_scorer.calculate_size_perf_ratio(0, 1_000_000, 'dense') == size_params['base_points']
    # Very high benchmark score should be capped at max_points
    assert edge_scorer.calculate_size_perf_ratio(100, 1_000_000, 'dense') <= size_params['max_points']
    # Test with None values
    assert edge_scorer.calculate_size_perf_ratio(None, 1, 'dense') == 0.0
    assert edge_scorer.calculate_size_perf_ratio(1, None, 'dense') == 0.0
    assert edge_scorer.calculate_size_perf_ratio(1, 1, None) == 0.0 
