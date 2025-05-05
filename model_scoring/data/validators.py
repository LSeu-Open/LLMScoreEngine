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
Data validation utilities for model data.

This module provides validation mechanisms for model data, ensuring that
the data conforms to the expected structure and constraints.
"""

from typing import Dict, Any, Optional
import logging

from ..core.constants import SCORE_BOUNDS, REQUIRED_SECTIONS, SCORE_SCALE, COMMUNITY_SCORE_BOUNDS
from ..core.exceptions import BenchmarkScoreError, ModelSpecificationError, CommunityScoreError, ModelDataValidationError

logger = logging.getLogger(__name__)

class ModelDataValidator:
    """Class to handle model data validation"""
    
    @staticmethod
    def validate_benchmarks(data: Dict, section: str, model_name: str) -> None:
        """
        Validates benchmark scores for a specific section of model data.

        Args:
            data (Dict): The model data dictionary containing benchmark scores
            section (str): The section name to validate (e.g. 'entity_benchmarks', 'dev_benchmarks')
            model_name (str): Name of the model being validated

        Raises:
            BenchmarkScoreError: If any validation check fails:
                - Section is not a dictionary
                - Required benchmark is missing
                - Score is not a number
                - Score is outside valid bounds

        Notes:
            - All scores are normalized by dividing by SCORE_SCALE
            - Valid scores must be between SCORE_BOUNDS["MIN"] and SCORE_BOUNDS["MAX"]
            - Scores can be None, which indicates benchmark was not run
        """
        # Verify section exists and is a dictionary
        if not isinstance(data[section], dict):
            raise BenchmarkScoreError(
                f"Section '{section}' must be a dictionary in model '{model_name}'"
            )
        
        # Validate all required benchmarks are present and have valid scores
        for field in REQUIRED_SECTIONS[section]:
            # Check benchmark exists
            if field not in data[section]:
                raise BenchmarkScoreError(
                    f"Missing benchmark '{field}' in {section} for model '{model_name}'"
                )
            
            score = data[section][field]
            # Only validate non-null scores
            if score is not None:
                # Verify score is numeric
                if not isinstance(score, (int, float)):
                    raise BenchmarkScoreError(
                        f"Invalid score type for '{field}' in {section}: expected number, got {type(score).__name__}"
                    )
                
                # Verify score is within valid bounds
                if score < SCORE_BOUNDS["MIN"] or score > SCORE_BOUNDS["MAX"]:
                    raise BenchmarkScoreError(
                        f"Score for '{field}' in {section} must be between {SCORE_BOUNDS['MIN']} and {SCORE_BOUNDS['MAX']}, got {score}"
                    )
                
                # Normalize score
                data[section][field] = score/SCORE_SCALE

    @staticmethod
    def validate_model_specs(specs: Dict, model_name: str) -> None:
        """Validate technical specifications for a language model.
        
        This method performs validation checks on the model's technical specifications
        to ensure all required fields are present and valid.

        Args:
            specs (Dict): Dictionary containing the model's technical specifications
                Expected fields are defined in REQUIRED_SECTIONS['model_specs']
            model_name (str): Name of the model being validated

        Raises:
            ModelSpecificationError: If any of these validation checks fail:
                - Required specification field is missing
                - Specification value is not a number (int/float)
                - Specification value is not positive (>0)

        Notes:
            - All specification values must be positive numbers
            - Common specs include parameters like model size, context length, etc.
            - The specific required fields are defined in REQUIRED_SECTIONS['model_specs']
        """
        # Validate each required specification field
        for field in REQUIRED_SECTIONS['model_specs']:
            # Check if required field exists
            if field not in specs:
                raise ModelSpecificationError(
                    f"Missing specification '{field}' in model_specs for '{model_name}'"
                )
            
            # Verify specification value is numeric
            if not isinstance(specs[field], (int, float)):
                raise ModelSpecificationError(
                    f"Invalid type for '{field}' in model_specs: expected number, got {type(specs[field]).__name__}"
                )
            
            # Ensure specification value is positive
            if specs[field] <= 0:
                raise ModelSpecificationError(
                    f"Specification '{field}' must be positive, got {specs[field]}"
                )

    @staticmethod
    def validate_community_score(score: Any, model_name: str) -> None:
        """Validate the community score for a language model.
        
        This method performs validation checks on a model's community score to ensure
        it meets the required criteria for validity.

        Args:
            score (Any): The community score value to validate. Must be a number.
            model_name (str): Name of the model being validated, used in error messages.

        Raises:
            CommunityScoreError: If any of these validation checks fail:
                - Score is not a numeric type (int/float)
                - Score is negative
                - Score is below minimum threshold (COMMUNITY_SCORE_BOUNDS["MIN"])
                - Score exceeds maximum threshold (COMMUNITY_SCORE_BOUNDS["MAX"])

        Notes:
            - Community scores represent the model's rating/reputation in the community
            - Valid scores must be positive numbers within defined bounds
            - Bounds are defined in COMMUNITY_SCORE_BOUNDS constants
        """
        # Validate score is numeric type
        if not isinstance(score, (int, float)):
            raise CommunityScoreError(
                f"Invalid community score type for '{model_name}': expected number, got {type(score).__name__}"
            )

        # Ensure score is not negative
        if score < 0:
            raise CommunityScoreError(
                f"Community score must be positive, got {score}"
            )

        # Validate score meets minimum threshold
        if score < COMMUNITY_SCORE_BOUNDS["MIN"]:
            raise CommunityScoreError(
                f"Community score must be at least {COMMUNITY_SCORE_BOUNDS['MIN']}, got {score}"
            )

        # Validate score does not exceed maximum threshold  
        if score > COMMUNITY_SCORE_BOUNDS["MAX"]:
            raise CommunityScoreError(
                f"Community score must be less than {COMMUNITY_SCORE_BOUNDS['MAX']}, got {score}"
            )


def validate_model_data(data: Dict, model_name: str) -> None:
    """
    Validate all model data for a given model.
    
    Args:
        data (Dict): The model data dictionary to validate
        model_name (str): Name of the model being validated
        
    Raises:
        ModelDataValidationError: If any validation check fails
    """
    validator = ModelDataValidator()
    
    # Verify all required sections exist
    for section in REQUIRED_SECTIONS:
        if section not in data:
            raise ModelDataValidationError(
                f"Missing required section '{section}' in model data for '{model_name}'"
            )

    # Validate benchmarks
    validator.validate_benchmarks(data, 'entity_benchmarks', model_name)
    validator.validate_benchmarks(data, 'dev_benchmarks', model_name)
    
    # Validate model specs
    validator.validate_model_specs(data['model_specs'], model_name)
    
    # Validate community score
    validator.validate_community_score(data['community_score'], model_name) 