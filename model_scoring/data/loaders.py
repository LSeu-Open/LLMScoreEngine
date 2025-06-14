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
Data loading utilities for model data.

This module provides functions for loading model data from JSON files.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import os
import json
import logging
from typing import Dict, Optional, Any
from types import ModuleType

from ..core.constants import MODELS_DIR
from ..core.types import ModelData
from .validators import validate_model_data

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------

def find_model_file(model_name: str, models_directory: str = MODELS_DIR) -> Optional[str]:
    """Locate the JSON file for a given model in the models directory.
    
    This function searches for a model's JSON file using the following strategy:
    1. First checks for an exact match with the model name
    2. If not found, tries a case-insensitive search
    
    Args:
        model_name (str): Name of the model to find the JSON file for
        models_directory (str): Directory path where model JSON files are stored
        
    Returns:
        Optional[str]: Full path to the model's JSON file if found, None otherwise
        
    Notes:
        - Expected JSON filename format is "{model_name}.json"
        - Search is case-insensitive as a fallback
        - All errors are logged before returning None
    """
    # First verify the models directory exists
    if not os.path.exists(models_directory):
        logger.error(f"Models directory '{models_directory}' not found")
        return None

    target_exact = f"{model_name}.json"
    target_lower = f"{model_name.lower()}.json"
    
    found_filename = None
    
    # Get all files to avoid repeated os calls and handle casing correctly
    try:
        filenames = os.listdir(models_directory)
    except OSError as e:
        logger.error(f"Could not read models directory '{models_directory}': {e}")
        return None

    # First, try to find an exact (case-sensitive) match
    for name in filenames:
        if name == target_exact:
            found_filename = name
            break  # Found the best match, no need to look further

    # If no exact match, try a case-insensitive search
    if not found_filename:
        for name in filenames:
            if name.lower() == target_lower:
                found_filename = name
                break  # Found a fallback match

    if found_filename:
        return os.path.join(models_directory, found_filename)

    # No matching file found after both attempts
    logger.error(f"No JSON file found for model '{model_name}'")
    return None

def load_json_file(file_path: str) -> Optional[Dict]:
    """Load and parse JSON file.
    
    Args:
        file_path (str): Path to the JSON file to load
        
    Returns:
        Optional[Dict]: Parsed JSON data as a dictionary if successful, None otherwise
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON format in '{file_path}': {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error loading JSON file '{file_path}': {str(e)}")
        return None

def load_model_data(model_name: str, models_directory: str = MODELS_DIR, 
                    scoring_config: Optional[ModuleType] = None) -> Optional[ModelData]:
    """Find, load and validate model data from a JSON file.

    This function handles the complete process of loading model data:
    1. Locates the model's JSON file
    2. Loads and parses the JSON data
    3. Validates the data structure and contents
    
    Args:
        model_name (str): Name of the model to load data for
        models_directory (str, optional): Directory containing model JSON files. 
            Defaults to MODELS_DIR constant.
        scoring_config (ModuleType, optional): A loaded configuration module.

    Returns:
        Optional[ModelData]: The loaded and validated model data dictionary if successful,
            None if any step fails (file not found, invalid JSON, validation error)

    Notes:
        - JSON files should be named "{model_name}.json"
        - File search is case-insensitive
        - All validation errors are logged before returning None
    """
    try:
        # First try to locate the model's JSON file
        json_file = find_model_file(model_name, models_directory)
        if not json_file:
            return None

        # Load and parse the JSON data
        data = load_json_file(json_file)
        if not data:
            return None

        # Validate the loaded data structure and contents
        validate_model_data(data, model_name, scoring_config=scoring_config)
        
        logger.info(f"Successfully validated data for model '{model_name}'")
        return data

    except Exception as e:
        # Catch and log any unexpected errors
        logger.error(f"Error processing model '{model_name}': {str(e)}")
        return None 
