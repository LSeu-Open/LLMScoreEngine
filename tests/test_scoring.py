"""Tests for the scoring module."""

import unittest
from model_scoring.scoring.models_scoring import ModelScorer

class TestModelScorer(unittest.TestCase):
    """Test case for the ModelScorer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.scorer = ModelScorer("Test Model")
    
    def test_entity_benchmarks(self):
        """Test entity benchmarks scoring."""
        # Perfect scores
        perfect_benchmarks = {
            'artificial_analysis': 1.0,
            'live_code_bench': 1.0,
            'big_code_models': 1.0,
            'open_llm': 1.0
        }
        
        perfect_score = self.scorer.calculate_entity_benchmarks(perfect_benchmarks)
        self.assertAlmostEqual(perfect_score, 25.0)
        
        # Empty benchmarks
        empty_score = self.scorer.calculate_entity_benchmarks({})
        self.assertEqual(empty_score, 0)
        
        # Partial benchmarks
        partial_benchmarks = {
            'artificial_analysis': 0.8,
            'live_code_bench': 0.6,
            'big_code_models': 0.9,
            'open_llm': 0.7
        }
        
        partial_score = self.scorer.calculate_entity_benchmarks(partial_benchmarks)
        self.assertAlmostEqual(partial_score, 0.75 * 25)  # 75% * 25 points
    
    def test_community_score(self):
        """Test community score calculation."""
        # Test minimum score
        min_score = self.scorer.calculate_community_score(1000)
        self.assertAlmostEqual(min_score, 0)
        
        # Test maximum score
        max_score = self.scorer.calculate_community_score(1402)
        self.assertAlmostEqual(max_score, 20)
        
        # Test middle score
        mid_elo = 1201  # 50% of the way from 1000 to 1402
        mid_score = self.scorer.calculate_community_score(mid_elo)
        self.assertAlmostEqual(mid_score, 10, delta=1)  # About 10 points (50% of 20)
    
    def test_technical_score(self):
        """Test technical score calculation."""
        # Best possible scores
        best_score = self.scorer.calculate_technical_score(
            price=0.5,  # < 1 = 8 points
            context_window=250000,  # > 200000 = 6 points
            size_perf_ratio=95  # > 90 = 6 points
        )
        self.assertEqual(best_score, 20)  # 8 + 6 + 6
        
        # Middle range scores
        mid_score = self.scorer.calculate_technical_score(
            price=15,  # < 20 = 4 points
            context_window=32001,  # > 32000 = 4 points
            size_perf_ratio=75  # > 70 = 4 points
        )
        self.assertEqual(mid_score, 12)  # 4 + 4 + 4
    
    def test_final_score(self):
        """Test final score calculation."""
        # Set component scores
        self.scorer.external_score = 50.5
        self.scorer.community_score = 15.8
        self.scorer.technical_score = 17.2
        
        # Calculate final score
        final_score = self.scorer.calculate_final_score()
        
        # The final score should be the sum of component scores, rounded to 2 decimals
        expected_score = round(50.5 + 15.8 + 17.2, 2)
        self.assertEqual(final_score, expected_score)
        
    def test_size_perf_ratio(self):
        """Test size/performance ratio calculation."""
        # High performance, small model = best efficiency
        high_perf_small = self.scorer.calculate_size_perf_ratio(90, 10)
        self.assertEqual(high_perf_small, 6.0)
        
        # High performance, large model = limited by model size
        high_perf_large = self.scorer.calculate_size_perf_ratio(90, 80)
        self.assertEqual(high_perf_large, 3.0)
        
        # Low performance, any size = limited by performance
        low_perf = self.scorer.calculate_size_perf_ratio(50, 10)
        self.assertEqual(low_perf, 2.0)

if __name__ == '__main__':
    unittest.main() 