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

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

from typing import Dict, Any, Optional
import logging
from types import ModuleType
import sys
from pathlib import Path
from model_scoring.core.constants import REQUIRED_SECTIONS
from model_scoring.core.exceptions import BenchmarkScoreError, ModelSpecificationError, CommunityScoreError, ModelDataValidationError
from config import scoring_config as default_scoring_config

# Add project root to sys.path for absolute imports.
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ------------------------------------------------------------------------------------------------
# Validation class and functions
# ------------------------------------------------------------------------------------------------

logger = logging.getLogger(__name__)

class ModelDataValidator:
    """Class to handle model data validation"""
    
    def __init__(self, scoring_config: ModuleType):
        """Initializes the validator with a scoring configuration."""
        self.config = scoring_config

    def validate_benchmarks(self, data: Dict, section: str, model_name: str) -> None:
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
            - Valid scores must be between SCORE_BOUNDS["min"] and SCORE_BOUNDS["max"]
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
                if score < self.config.SCORE_BOUNDS["min"] or score > self.config.SCORE_BOUNDS["max"]:
                    raise BenchmarkScoreError(
                        f"Score for '{field}' in {section} must be between {self.config.SCORE_BOUNDS['min']} and {self.config.SCORE_BOUNDS['max']}, got {score}"
                    )
                
                # Normalize score
                data[section][field] = score / self.config.SCORE_SCALE

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
                - Specification value is not of the expected type (str for architecture, number for others)
                - Numeric specification value is not positive (>0)

        Notes:
            - 'architecture' field must be a string.
            - Other specification values must be positive numbers.
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
            
            # Handle 'architecture' field specifically (must be string)
            if field == 'architecture':
                if not isinstance(specs[field], str):
                    raise ModelSpecificationError(
                        f"Invalid type for 'architecture' in model_specs for '{model_name}': expected str, got {type(specs[field]).__name__}"
                    )
                if not specs[field].strip(): # Ensure architecture string is not empty or just whitespace
                     raise ModelSpecificationError(
                        f"Specification 'architecture' cannot be empty in model_specs for '{model_name}'"
                    )
                continue # Skip numeric checks for architecture

            # Verify other specification values are numeric
            if not isinstance(specs[field], (int, float)):
                raise ModelSpecificationError(
                    f"Invalid type for '{field}' in model_specs for '{model_name}': expected number, got {type(specs[field]).__name__}"
                )
            
            # Ensure numeric specification value is positive
            if specs[field] <= 0:
                raise ModelSpecificationError(
                    f"Specification '{field}' must be positive for '{model_name}', got {specs[field]}"
                )

    def validate_community_score(self, scores_data: Dict[str, Any], model_name: str) -> None:
        """Validate the community scores for a language model.
        
        This method checks that the community_score section is a dictionary
        and validates its constituent scores (e.g., lm_sys_arena_score, hf_score)
        against their respective bounds (0-10 points each).

        Args:
            scores_data (Dict[str, Any]): The dictionary containing community scores.
                                          Expected to have keys as defined in REQUIRED_SECTIONS['community_score'].
            model_name (str): Name of the model being validated, used in error messages.

        Raises:
            CommunityScoreError: If any of these validation checks fail:
                - community_score section is not a dictionary.
                - A required score field (e.g., 'lm_sys_arena_score', 'hf_score') is missing.
                - A score value is not a numeric type (int/float) when not None.
                - A score value is outside its defined bounds (e.g., 0-10).

        Notes:
            - Individual scores can be None, indicating the score is not yet available.
            - Bounds for each score component are defined in constants.py (e.g., LM_SYS_ARENA_SCORE_BOUNDS).
        """
        if not isinstance(scores_data, dict):
            raise CommunityScoreError(
                f"Community score section must be a dictionary for model '{model_name}', got {type(scores_data).__name__}"
            )

        for field in REQUIRED_SECTIONS['community_score']:
            if field not in scores_data:
                raise CommunityScoreError(
                    f"Missing community score field '{field}' for model '{model_name}'"
                )
            
            score_value = scores_data[field]

            # Only validate non-null scores
            if score_value is not None:
                if not isinstance(score_value, (int, float)):
                    raise CommunityScoreError(
                        f"Invalid type for community score '{field}' for model '{model_name}': expected number, got {type(score_value).__name__}"
                    )

                if field in self.config.COMMUNITY_SCORE_BOUNDS:
                    bounds = self.config.COMMUNITY_SCORE_BOUNDS[field]
                    if not (bounds["min"] <= score_value <= bounds["max"]):
                        raise CommunityScoreError(
                            f"Community score for '{field}' for model '{model_name}' must be between {bounds['min']} and {bounds['max']}, got {score_value}"
                        )
                else:
                    # This case should not be reached if REQUIRED_SECTIONS['community_score'] is well-defined
                    # and corresponding bounds constants exist for each field.
                    logger.warning(
                        f"No defined bounds for community score field '{field}' for model '{model_name}'. Skipping bounds check."
                    )


def validate_model_data(data: Dict, model_name: str, scoring_config: Optional[ModuleType] = None) -> None:
    """
    Validate all model data for a given model.
    
    Args:
        data (Dict): The model data dictionary to validate
        model_name (str): Name of the model being validated
        scoring_config (ModuleType, optional): A loaded configuration module.
        
    Raises:
        ModelDataValidationError: If any validation check fails
    """
    config = scoring_config if scoring_config is not None else default_scoring_config
    validator = ModelDataValidator(scoring_config=config)
    
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
