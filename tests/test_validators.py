"""Tests for the data validators."""

import unittest
from model_scoring.core.exceptions import (
    BenchmarkScoreError, ModelSpecificationError, CommunityScoreError, ModelDataValidationError
)
from model_scoring.data.validators import ModelDataValidator

class TestModelDataValidator(unittest.TestCase):
    """Test case for the ModelDataValidator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.validator = ModelDataValidator()
        self.model_name = "Test Model"
    
    def test_validate_benchmarks_valid(self):
        """Test that valid benchmark data passes validation."""
        # Valid benchmark data
        data = {
            'entity_benchmarks': {
                'artificial_analysis': 80,
                'live_code_bench': 75,
                'big_code_models': 90,
                'open_llm': 85
            }
        }
        
        # Should not raise an exception
        try:
            self.validator.validate_benchmarks(data, 'entity_benchmarks', self.model_name)
        except BenchmarkScoreError:
            self.fail("validate_benchmarks() raised BenchmarkScoreError unexpectedly!")
    
    def test_validate_benchmarks_invalid_score(self):
        """Test that benchmark data with invalid scores fails validation."""
        # Invalid score (negative value)
        data = {
            'entity_benchmarks': {
                'artificial_analysis': -10,  # Invalid
                'live_code_bench': 75,
                'big_code_models': 90,
                'open_llm': 85
            }
        }
        
        # Should raise an exception
        with self.assertRaises(BenchmarkScoreError):
            self.validator.validate_benchmarks(data, 'entity_benchmarks', self.model_name)
    
    def test_validate_model_specs_valid(self):
        """Test that valid model specifications pass validation."""
        # Valid model specs
        specs = {
            'price': 10.0,
            'context_window': 32000,
            'param_count': 70.0
        }
        
        # Should not raise an exception
        try:
            self.validator.validate_model_specs(specs, self.model_name)
        except ModelSpecificationError:
            self.fail("validate_model_specs() raised ModelSpecificationError unexpectedly!")
    
    def test_validate_model_specs_missing_field(self):
        """Test that model specs with missing fields fail validation."""
        # Missing 'price' field
        specs = {
            'context_window': 32000,
            'param_count': 70.0
        }
        
        # Should raise an exception
        with self.assertRaises(ModelSpecificationError):
            self.validator.validate_model_specs(specs, self.model_name)
    
    def test_validate_community_score_valid(self):
        """Test that valid community scores pass validation."""
        # Valid score
        score = 1200
        
        # Should not raise an exception
        try:
            self.validator.validate_community_score(score, self.model_name)
        except CommunityScoreError:
            self.fail("validate_community_score() raised CommunityScoreError unexpectedly!")
    
    def test_validate_community_score_too_low(self):
        """Test that community scores that are too low fail validation."""
        # Score below minimum threshold
        score = 800
        
        # Should raise an exception
        with self.assertRaises(CommunityScoreError):
            self.validator.validate_community_score(score, self.model_name)

if __name__ == '__main__':
    unittest.main()