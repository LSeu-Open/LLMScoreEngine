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
Tests for the configuration loader utility.
"""

import pytest
from types import ModuleType
from model_scoring.utils.config_loader import load_config_from_path

def test_load_config_from_path_success(tmp_path):
    """Test that a valid configuration file is loaded successfully."""
    # Arrange: Create a temporary valid config file
    config_content = 'TEST_VAR = 123\nANOTHER_VAR = "hello"'
    config_file = tmp_path / "good_config.py"
    config_file.write_text(config_content)
    
    # Act: Load the configuration
    loaded_config = load_config_from_path(str(config_file))
    
    # Assert: Check that it's a module with the correct attributes
    assert isinstance(loaded_config, ModuleType)
    assert hasattr(loaded_config, 'TEST_VAR')
    assert loaded_config.TEST_VAR == 123
    assert hasattr(loaded_config, 'ANOTHER_VAR')
    assert loaded_config.ANOTHER_VAR == "hello"

def test_load_config_not_found():
    """Test that FileNotFoundError is raised for a non-existent file."""
    # Arrange: Define a path to a file that doesn't exist
    non_existent_path = "path/to/non_existent_config.py"
    
    # Act & Assert: Expect a FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Configuration file not found at:"):
        load_config_from_path(non_existent_path)

def test_load_config_with_invalid_syntax(tmp_path):
    """Test that an exception is raised for a file with invalid Python syntax."""
    # Arrange: Create a temporary config file with a syntax error
    config_content = "THIS IS NOT VALID PYTHON ="
    config_file = tmp_path / "bad_config.py"
    config_file.write_text(config_content)
    
    # Act & Assert: Expect an exception during module execution
    with pytest.raises(Exception, match="Failed to load configuration from"):
        load_config_from_path(str(config_file)) 