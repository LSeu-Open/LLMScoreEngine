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

# This file contains unit tests for the ModelScorer class

# ------------------------------------------------------------------------------------------------

import unittest
from model_scoring.scoring.models_scoring import ModelScorer

class TestModelScorer(unittest.TestCase):
    def setUp(self):
        self.scorer = ModelScorer()
        
    def test_entity_benchmarks(self):
        print("\n=== Testing Entity Benchmarks ===")
        # Test perfect scores
        perfect_benchmarks = {
            'artificial_analysis': 1.0,
            'live_code_bench': 1.0,
            'big_code_models': 1.0,
            'open_vlm': 1.0,
            'open_llm': 1.0,
            'mm_bench': 1.0
        }
        perfect_score = self.scorer.calculate_entity_benchmarks(perfect_benchmarks)
        print(f"Perfect score test: {perfect_score}/25.0")
        self.assertAlmostEqual(perfect_score, 25.0)
        
        # Test empty benchmarks
        empty_score = self.scorer.calculate_entity_benchmarks({})
        print(f"Empty benchmarks test: {empty_score}")
        self.assertEqual(empty_score, 0)
        
        # Test partial benchmarks (some None values)
        partial_benchmarks = {
            'artificial_analysis': 0.8,
            'live_code_bench': None,
            'big_code_models': 0.75,
            'open_vlm': 0.9
        }
        partial_score = self.scorer.calculate_entity_benchmarks(partial_benchmarks)
        print(f"Partial benchmarks test: {partial_score}/25.0")
        self.assertTrue(0 < partial_score < 25)

    def test_dev_benchmarks(self):
        print("\n=== Testing Dev Benchmarks ===")
        # Test perfect scores
        perfect_dev = {k: 1.0 for k in [
            'MMLU', 'MMLU Pro', 'Multilingual MMLU', 'BFCL',
            'HumanEval', 'MBPP', 'Gorilla Benchmark'
        ]}
        perfect_score = self.scorer.calculate_dev_benchmarks(perfect_dev)
        print(f"Perfect score test: {perfect_score}/35.0")
        self.assertAlmostEqual(perfect_score, 35.0)
        
        # Test empty benchmarks
        empty_score = self.scorer.calculate_dev_benchmarks({})
        print(f"Empty benchmarks test: {empty_score}")
        self.assertEqual(empty_score, 0)
        
        # Test invalid benchmark name
        invalid_bench = {'InvalidBenchmark': 1.0}
        invalid_score = self.scorer.calculate_dev_benchmarks(invalid_bench)
        print(f"Invalid benchmark test: {invalid_score}")
        self.assertEqual(invalid_score, 0)

    def test_community_score(self):
        print("\n=== Testing Community Score ===")
        # Test minimum score
        min_score = self.scorer.calculate_community_score(1000)
        print(f"Minimum score test (ELO 1000): {min_score}")
        self.assertAlmostEqual(min_score, 0)
        
        # Test maximum score
        max_score = self.scorer.calculate_community_score(1365)
        print(f"Maximum score test (ELO 1365): {max_score}")
        self.assertAlmostEqual(max_score, 20)
        
        # Test middle score
        mid_score = self.scorer.calculate_community_score(1182.5)  # halfway
        print(f"Middle score test (ELO 1182.5): {mid_score}")
        self.assertTrue(9.5 < mid_score < 10.5)
        
        # Test None input
        none_score = self.scorer.calculate_community_score(None)
        print(f"None input test: {none_score}")
        self.assertIsNone(none_score)

    def test_technical_score(self):
        print("\n=== Testing Technical Score ===")
        # Test best possible scores
        best_score = self.scorer.calculate_technical_score(
            price=0.5,  # < 1
            context_window=250000,  # > 200000
            size_perf_ratio=95  # > 90
        )
        print(f"Best possible scores test: {best_score}/20")
        self.assertEqual(best_score, 20)  # 8 + 6 + 6
        
        # Test worst possible scores
        worst_score = self.scorer.calculate_technical_score(
            price=100,  # > 80
            context_window=4000,  # < 8000
            size_perf_ratio=50  # < 60
        )
        print(f"Worst possible scores test: {worst_score}/20")
        self.assertEqual(worst_score, 4)  # 1 + 1 + 2

    def test_final_score(self):
        print("\n=== Testing Final Score ===")
        # Test perfect scores
        self.scorer.external_score = 60
        self.scorer.community_score = 20
        self.scorer.technical_score = 20
        perfect_score = self.scorer.calculate_final_score()
        print(f"Perfect scores test: {perfect_score}/100")
        self.assertEqual(perfect_score, 100)
        
        # Test missing scores
        print("Testing missing scores...")
        new_scorer = ModelScorer()
        with self.assertRaises(ValueError):
            new_scorer.calculate_final_score()
        print("Missing scores test: Correctly raised ValueError")

    def test_weight_redistribution_entity(self):
        print("\n=== Testing Entity Benchmark Weight Redistribution ===")
        
        original_weights = {
            'artificial_analysis': 16.67,
            'live_code_bench': 16.67,
            'big_code_models': 16.67,
            'open_vlm': 16.67,
            'open_llm': 16.67,
            'mm_bench': 16.67
        }
        print("\nOriginal weights:")
        for bench, weight in original_weights.items():
            print(f"{bench}: {weight:.2f}")
        
        # Test with one benchmark missing
        one_missing = {
            'artificial_analysis': 1.0,
            'live_code_bench': None,  # Missing
            'big_code_models': 1.0,
            'open_vlm': 1.0,
            'open_llm': 1.0,
            'mm_bench': 1.0
        }
        available_weight = sum(original_weights[k] for k, v in one_missing.items() if v is not None)
        print("\nRedistributed weights with one missing:")
        for bench, score in one_missing.items():
            if score is not None:
                redistributed_weight = (original_weights[bench] / available_weight) * 100
                print(f"{bench}: {redistributed_weight:.2f} (original: {original_weights[bench]:.2f})")
        one_missing_score = self.scorer.calculate_entity_benchmarks(one_missing)
        print(f"Score with one missing benchmark: {one_missing_score:.2f}/25.0")

        # Test with multiple benchmarks missing
        multiple_missing = {
            'artificial_analysis': 1.0,
            'live_code_bench': None,
            'big_code_models': None,
            'open_vlm': 1.0,
            'open_llm': None,
            'mm_bench': 1.0
        }
        available_weight = sum(original_weights[k] for k, v in multiple_missing.items() if v is not None)
        print("\nRedistributed weights with multiple missing:")
        for bench, score in multiple_missing.items():
            if score is not None:
                redistributed_weight = (original_weights[bench] / available_weight) * 100
                print(f"{bench}: {redistributed_weight:.2f} (original: {original_weights[bench]:.2f})")
        multiple_missing_score = self.scorer.calculate_entity_benchmarks(multiple_missing)
        print(f"Score with multiple missing benchmarks: {multiple_missing_score:.2f}/25.0")

    def test_weight_redistribution_dev(self):
        print("\n=== Testing Dev Benchmark Weight Redistribution ===")
        
        original_weights = {
            'MMLU': 3.5,
            'MMLU Pro': 5.5,
            'Multilingual MMLU': 3.5,
            'IFEval': 3.5,
            'Arena-Hard': 3.5,
            'GPQA': 3.5,
            'ARC-C': 3.5,
            'BigBench': 3.5,
            'TruthfulQA': 3.5,
            'AlignBench': 5.5,
            'Wild Bench': 3.5,
            'MT-bench': 3.5,
            'MATH': 3.5,
            'GSM-8K': 3.5,
            'HumanEval': 3.5,
            'HumanEval Plus': 3.5,
            'MBPP': 3.5,
            'MBPP Plus': 3.5,
            'SWE-bench': 3.5,
            'API-Bank': 3.5,
            'BFCL': 10.0,
            'Gorilla Benchmark': 3.5,
            'Nexus': 3.5
        }
        
        total_weight = sum(original_weights.values())
        print("\nOriginal weights (total weight: {:.1f}):".format(total_weight))
        for bench, weight in original_weights.items():
            percentage = (weight / total_weight) * 100
            print(f"{bench}: {weight:.1f} ({percentage:.1f}% of total)")

        # Test Scenario 1: Only high-weight benchmarks present
        print("\nScenario 1: Only high-weight benchmarks present")
        high_weight_only = {
            'MMLU Pro': 0.9,    # 5.5
            'AlignBench': 0.85, # 5.5
            'BFCL': 0.95,       # 10.0
        }
        available_weight = sum(original_weights[k] for k in high_weight_only.keys())
        print(f"Available weight: {available_weight:.1f} out of {total_weight:.1f}")
        for bench, score in high_weight_only.items():
            redistributed_weight = (original_weights[bench] / available_weight) * 100
            print(f"{bench}: {redistributed_weight:.2f}% (original: {original_weights[bench]:.1f})")
        high_weight_score = self.scorer.calculate_dev_benchmarks(high_weight_only)
        print(f"Score: {high_weight_score:.2f}/35.0")

        # Test Scenario 2: Only standard benchmarks present
        print("\nScenario 2: Only standard benchmarks present")
        standard_only = {
            'MMLU': 0.8,
            'GSM-8K': 0.85,
            'HumanEval': 0.75,
            'MBPP': 0.9
        }
        available_weight = sum(original_weights[k] for k in standard_only.keys())
        print(f"Available weight: {available_weight:.1f} out of {total_weight:.1f}")
        for bench, score in standard_only.items():
            redistributed_weight = (original_weights[bench] / available_weight) * 100
            print(f"{bench}: {redistributed_weight:.2f}% (original: {original_weights[bench]:.1f})")
        standard_score = self.scorer.calculate_dev_benchmarks(standard_only)
        print(f"Score: {standard_score:.2f}/35.0")

        # Test Scenario 3: Mix of high and standard weights with some missing
        print("\nScenario 3: Mix of high and standard weights")
        mixed_scenario = {
            'MMLU': 0.8,        # 3.5
            'MMLU Pro': 0.9,    # 5.5
            'BFCL': 0.95,       # 10.0
            'HumanEval': 0.75,  # 3.5
            'MBPP': 0.9,        # 3.5
            'AlignBench': None, # 5.5 (missing)
            'GSM-8K': None      # 3.5 (missing)
        }
        available_weight = sum(original_weights[k] for k, v in mixed_scenario.items() if v is not None)
        print(f"Available weight: {available_weight:.1f} out of {total_weight:.1f}")
        for bench, score in mixed_scenario.items():
            if score is not None:
                redistributed_weight = (original_weights[bench] / available_weight) * 100
                print(f"{bench}: {redistributed_weight:.2f}% (original: {original_weights[bench]:.1f})")
                print(f"  Score: {score:.2f} * Weight: {redistributed_weight:.2f}% = {score * redistributed_weight:.2f}")
        mixed_score = self.scorer.calculate_dev_benchmarks(mixed_scenario)
        print(f"Score: {mixed_score:.2f}/35.0")

        # Test Scenario 4: Missing all high-weight benchmarks
        print("\nScenario 4: Missing all high-weight benchmarks")
        no_high_weight = {k: 0.8 for k in original_weights.keys() 
                         if k not in ['MMLU Pro', 'AlignBench', 'BFCL']}
        available_weight = sum(original_weights[k] for k in no_high_weight.keys())
        print(f"Available weight: {available_weight:.1f} out of {total_weight:.1f}")
        print(f"Missing weights: MMLU Pro (5.5), AlignBench (5.5), BFCL (10.0)")
        no_high_score = self.scorer.calculate_dev_benchmarks(no_high_weight)
        print(f"Score: {no_high_score:.2f}/35.0")

        # Test Scenario 5: Real-world typical scenario
        print("\nScenario 5: Real-world typical scenario")
        typical_scenario = {
            'MMLU': 0.82,           # Common
            'MMLU Pro': 0.85,       # High weight
            'HumanEval': 0.75,      # Coding
            'MBPP': 0.90,           # Coding
            'BFCL': 0.95,          # High weight
            'GSM-8K': 0.82,         # Math
            'MATH': 0.75,           # Math
            'TruthfulQA': 0.82,     # Safety
            'AlignBench': None,     # Missing high weight
            'API-Bank': None,       # Missing
            'SWE-bench': None,      # Missing
            'Nexus': None           # Missing
        }
        available_weight = sum(original_weights[k] for k, v in typical_scenario.items() if v is not None)
        print(f"Available weight: {available_weight:.1f} out of {total_weight:.1f}")
        for bench, score in typical_scenario.items():
            if score is not None:
                redistributed_weight = (original_weights[bench] / available_weight) * 100
                print(f"{bench}: {redistributed_weight:.2f}% (original: {original_weights[bench]:.1f})")
                print(f"  Score: {score:.2f} * Weight: {redistributed_weight:.2f}% = {score * redistributed_weight:.2f}")
        typical_score = self.scorer.calculate_dev_benchmarks(typical_scenario)
        print(f"Score: {typical_score:.2f}/35.0")

    def test_score_calculations(self):
        print("\n=== Testing Score Calculations ===")
        
        # Test Scenario 1: Complete benchmark set
        print("\nScenario 1: Complete benchmark set")
        entity_benchmarks = {
            'artificial_analysis': 0.82,
            'live_code_bench': 0.85,
            'big_code_models': 0.75,
            'open_vlm': 0.90,
            'open_llm': 0.85,
            'mm_bench': 0.88
        }
        
        dev_benchmarks = {
            'MMLU': 0.82,
            'MMLU Pro': 0.85,
            'Multilingual MMLU': 0.80,
            'IFEval': 0.75,
            'Arena-Hard': 0.78,
            'GPQA': 0.83,
            'ARC-C': 0.85,
            'BigBench': 0.80,
            'TruthfulQA': 0.82,
            'AlignBench': 0.85,
            'Wild Bench': 0.80,
            'MT-bench': 0.85,
            'MATH': 0.75,
            'GSM-8K': 0.82,
            'HumanEval': 0.75,
            'HumanEval Plus': 0.78,
            'MBPP': 0.90,
            'MBPP Plus': 0.88,
            'SWE-bench': 0.85,
            'API-Bank': 0.80,
            'BFCL': 0.95,
            'Gorilla Benchmark': 0.85,
            'Nexus': 0.82
        }
        
        # Calculate and print entity score
        entity_score = self.scorer.calculate_entity_benchmarks(entity_benchmarks)
        print("\nEntity Benchmarks Calculation:")
        total_entity_weight = 0
        weighted_entity_sum = 0
        for bench, score in entity_benchmarks.items():
            weight = 16.67  # Equal weights for entity benchmarks
            weighted_score = score * weight
            weighted_entity_sum += weighted_score
            total_entity_weight += weight
            print(f"{bench}: {score:.2f} * {weight:.2f} = {weighted_score:.2f}")
        final_entity_score = (weighted_entity_sum / total_entity_weight) * 25
        print(f"Entity Score: {final_entity_score:.2f}/25.0")
        
        # Calculate and print dev score
        dev_score = self.scorer.calculate_dev_benchmarks(dev_benchmarks)
        print("\nDev Benchmarks Calculation:")
        weights = {
            'MMLU': 3.5, 'MMLU Pro': 5.5, 'Multilingual MMLU': 3.5,
            'IFEval': 3.5, 'Arena-Hard': 3.5, 'GPQA': 3.5,
            'ARC-C': 3.5, 'BigBench': 3.5, 'TruthfulQA': 3.5,
            'AlignBench': 5.5, 'Wild Bench': 3.5, 'MT-bench': 3.5,
            'MATH': 3.5, 'GSM-8K': 3.5, 'HumanEval': 3.5,
            'HumanEval Plus': 3.5, 'MBPP': 3.5, 'MBPP Plus': 3.5,
            'SWE-bench': 3.5, 'API-Bank': 3.5, 'BFCL': 10.0,
            'Gorilla Benchmark': 3.5, 'Nexus': 3.5
        }
        total_dev_weight = 0
        weighted_dev_sum = 0
        for bench, score in dev_benchmarks.items():
            weight = weights[bench]
            weighted_score = score * weight
            weighted_dev_sum += weighted_score
            total_dev_weight += weight
            print(f"{bench}: {score:.2f} * {weight:.2f} = {weighted_score:.2f}")
        final_dev_score = (weighted_dev_sum / total_dev_weight) * 35
        print(f"Dev Score: {final_dev_score:.2f}/35.0")
        
        # Test Scenario 2: Technical Score Calculation
        print("\nScenario 2: Technical Score Calculation")
        technical_params = {
            'price': 2.5,
            'context_window': 100000,
            'size_perf_ratio': 85
        }
        
        print("\nTechnical Score Components:")
        price_score = self.scorer._calculate_price_score(technical_params['price'])
        print(f"Price Score ({technical_params['price']}$): {price_score}/8")
        
        context_score = self.scorer._calculate_context_score(technical_params['context_window'])
        print(f"Context Window Score ({technical_params['context_window']} tokens): {context_score}/6")
        
        ratio_score = self.scorer._calculate_ratio_score(technical_params['size_perf_ratio'])
        print(f"Size/Performance Ratio Score ({technical_params['size_perf_ratio']}): {ratio_score}/6")
        
        technical_score = self.scorer.calculate_technical_score(
            price=technical_params['price'],
            context_window=technical_params['context_window'],
            size_perf_ratio=technical_params['size_perf_ratio']
        )
        print(f"Total Technical Score: {technical_score}/20")
        
        # Test Scenario 3: Community Score Calculation
        print("\nScenario 3: Community Score Calculation")
        elo_ratings = [1000, 1182.5, 1365]  # min, mid, max
        print("\nCommunity Score at different ELO ratings:")
        for elo in elo_ratings:
            score = self.scorer.calculate_community_score(elo)
            normalized = ((elo - 1000) / (1365 - 1000)) * 100
            print(f"ELO {elo}: Normalized {normalized:.2f}% -> Score {score:.2f}/20")
        
        # Test Scenario 4: Final Score Calculation
        print("\nScenario 4: Final Score Calculation")
        self.scorer.external_score = entity_score + dev_score
        self.scorer.community_score = self.scorer.calculate_community_score(1250)
        self.scorer.technical_score = technical_score
        
        final_score = self.scorer.calculate_final_score()
        print("\nFinal Score Components:")
        print(f"External Score: {self.scorer.external_score:.2f}/60")
        print(f"Community Score: {self.scorer.community_score:.2f}/20")
        print(f"Technical Score: {self.scorer.technical_score}/20")
        print(f"Final Score: {final_score:.2f}/100")

    def test_size_performance_ratio(self):
        print("\n=== Testing Size/Performance Ratio ===")
        
        test_cases = [
            # Very Large Models (≥70B) - Capped at 3.0
            (95, 70, 3.0, "Excellent performance (95%), very large model (70B)"),
            (85, 70, 3.0, "Excellent performance (85%), very large model (70B)"),
            (75, 70, 3.0, "Good performance (75%), very large model (70B)"),
            (65, 70, 3.0, "Decent performance (65%), very large model (70B)"),
            (55, 70, 2.0, "Poor performance (55%), very large model (70B)"),
            
            # Large Models (40-70B) - Capped at 4.0
            (95, 45, 4.0, "Excellent performance (95%), large model (45B)"),
            (85, 45, 4.0, "Excellent performance (85%), large model (45B)"),
            (75, 45, 4.0, "Good performance (75%), large model (45B)"),
            (65, 45, 4.0, "Decent performance (65%), large model (45B)"),
            (55, 45, 2.0, "Moderate performance (55%), large model (45B)"),
            
            # Medium-Large Models (30-40B) - Capped at 5.0
            (95, 35, 5.0, "Excellent performance (95%), medium-large model (35B)"),
            (85, 35, 5.0, "Excellent performance (85%), medium-large model (35B)"),
            (75, 35, 5.0, "Good performance (75%), medium-large model (35B)"),
            (65, 35, 4.0, "Decent performance (65%), medium-large model (35B)"),
            (55, 35, 3.0, "Moderate performance (55%), medium-large model (35B)"),
            
            # Medium Models (15-30B) - Capped at 5.5
            (95, 20, 5.5, "Excellent performance (95%), medium model (20B)"),
            (85, 20, 5.5, "Excellent performance (85%), medium model (20B)"),
            (75, 20, 5.0, "Good performance (75%), medium model (20B)"),
            (65, 20, 4.0, "Decent performance (65%), medium model (20B)"),
            (55, 20, 3.0, "Moderate performance (55%), medium model (20B)"),
            
            # Small Models (<15B) - No cap
            (95, 7, 6.0, "Excellent performance (95%), small model (7B)"),
            (85, 7, 6.0, "Excellent performance (85%), small model (7B)"),
            (75, 7, 5.0, "Good performance (75%), small model (7B)"),
            (65, 7, 4.0, "Decent performance (65%), small model (7B)"),
            (55, 7, 3.0, "Moderate performance (55%), small model (7B)"),
            
            # Edge Cases
            (84, 7, 5.0, "Almost excellent (84%), small model (7B)"),
            (74, 7, 4.0, "Almost good (74%), small model (7B)"),
            (64, 7, 3.0, "Almost decent (64%), small model (7B)"),
            (54, 7, 2.0, "Almost moderate (54%), small model (7B)"),
            
            # Boundary Cases for Model Sizes
            (85, 14.9, 6.0, "Excellent performance (85%), just under medium size"),
            (85, 15.1, 5.5, "Excellent performance (85%), just over medium size"),
            (85, 29.9, 5.5, "Excellent performance (85%), just under medium-large"),
            (85, 30.1, 5.0, "Excellent performance (85%), just over medium-large"),
            (85, 39.9, 5.0, "Excellent performance (85%), just under large"),
            (85, 40.1, 4.0, "Excellent performance (85%), just over large"),
            (85, 69.9, 4.0, "Excellent performance (85%), just under very large"),
            (85, 70.1, 3.0, "Excellent performance (85%), just over very large")
        ]
        
        print("\nTesting all size/performance combinations:")
        for benchmark_score, params, expected, description in test_cases:
            ratio_score = self.scorer.calculate_size_perf_ratio(benchmark_score, params)
            
            # Print test case details in a more organized format
            print(f"\nTest Case: {description}")
            print("-" * 50)
            print(f"Input:")
            print(f"  Benchmark Score: {benchmark_score:>3}%")
            print(f"  Parameter Count: {params:>4}B")
            print(f"\nScores:")
            print(f"  Expected: {expected:>4.1f}/6.0")
            print(f"  Actual:   {ratio_score:>4.1f}/6.0")
            
            # Validate results
            self.assertAlmostEqual(
                ratio_score, 
                expected,
                places=1,
                msg=f"Score mismatch for {description}:\n" \
                    f"Expected {expected}, got {ratio_score}"
            )
            
            # Additional validation
            self.assertGreaterEqual(ratio_score, 0, "Score should not be negative")
            self.assertLessEqual(ratio_score, 6, "Score should not exceed 6.0")

    def test_small_models_size_performance(self):
        print("\n=== Testing Size/Performance Ratio for Small Models ===")
        
        small_model_cases = [
            (87, 1.5, 6.0, "Very small model (1.5B) with excellent performance")
        ]
        
        print("\nTesting various small model configurations:")
        for benchmark_score, params, expected, description in small_model_cases:
            ratio_score = self.scorer.calculate_size_perf_ratio(benchmark_score, params)
            print(f"\nScenario: {description}")
            print(f"Benchmark Score: {benchmark_score}%")
            print(f"Parameter Count: {params}B")
            print(f"Expected Score: {expected}/6.0")
            print(f"Actual Score: {ratio_score}/6.0")
            # Allow for small floating point differences
            self.assertEqual(ratio_score, expected,
                msg=f"Failed for {description}: expected {expected}, got {ratio_score}")
        
        # Test comparative scenarios
        print("\nComparative Scenarios:")
        
        # Compare similar performance at different sizes
        similar_perf_cases = [
            (85, 1.5, 5.0, "1.5B model with 85% performance"),
            (85, 7.0, 5.0, "7B model with 85% performance"),
            (85, 32.0, 5.0, "32B model with 85% performance"),
            (85, 70.0, 3.0, "70B model with 85% performance")
        ]
        
        print("\nModels with similar performance (85%) at different sizes:")
        for score, params, expected, description in similar_perf_cases:
            ratio_score = self.scorer.calculate_size_perf_ratio(score, params)
            print(f"\n{description}")
            print(f"Parameter Count: {params}B")
            print(f"Expected Score: {expected}/6.0")
            print(f"Actual Score: {ratio_score}/6.0")
            self.assertEqual(ratio_score, expected)
        
        # Compare different performances at similar sizes
        small_size_cases = [
            (92, 3.0, 6.0, "3B model with excellent performance (92%)"),
            (85, 3.0, 5.0, "3B model with good performance (85%)"),
            (75, 3.0, 4.0, "3B model with decent performance (75%)"),
            (65, 3.0, 3.0, "3B model with moderate performance (65%)"),
            (55, 3.0, 2.0, "3B model with poor performance (55%)")
        ]
        
        print("\nSame size (3B) with different performances:")
        for score, params, expected, description in small_size_cases:
            ratio_score = self.scorer.calculate_size_perf_ratio(score, params)
            print(f"\n{description}")
            print(f"Benchmark Score: {score}%")
            print(f"Parameter Count: {params}B")
            print(f"Expected Score: {expected}/6.0")
            print(f"Actual Score: {ratio_score}/6.0")
            self.assertEqual(ratio_score, expected,
                msg=f"Failed for {description}: expected {expected}, got {ratio_score}")
            # Verify score is within valid range
            self.assertGreaterEqual(ratio_score, 0, "Score should not be negative")
            self.assertLessEqual(ratio_score, 6, "Score should not exceed maximum of 6")
            # For models with same size, verify higher performance gets better score
            if score > 55:  # Compare with previous test case
                prev_score = self.scorer.calculate_size_perf_ratio(score-10, params)
                self.assertGreater(ratio_score, prev_score,
                    "Higher performance should result in better score for same size model")

if __name__ == '__main__':
    unittest.main(verbosity=2)