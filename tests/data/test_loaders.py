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
Tests for the data loading utilities.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import pytest
import json
import logging
from unittest.mock import patch

from model_scoring.data import loaders

# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------

@pytest.fixture
def models_dir(tmp_path):
    """Create a temporary models directory with mock model files."""
    models_path = tmp_path / "models"
    models_path.mkdir()
    
    # Valid model file (exact match)
    (models_path / "TestModel.json").write_text(json.dumps({"name": "TestModel", "architecture": "dense"}))
    
    # Valid model file (case-insensitive match)
    (models_path / "anothermodel.json").write_text(json.dumps({"name": "AnotherModel", "architecture": "moe"}))
    
    # Invalid JSON file
    (models_path / "InvalidJson.json").write_text("{'name': 'bad json',}")
    
    return str(models_path)

# ------------------------------------------------------------------------------------------------
# Tests for find_model_file
# ------------------------------------------------------------------------------------------------

def test_find_model_file_exact_match(models_dir):
    """Test finding a model file with an exact, case-sensitive match."""
    result = loaders.find_model_file("TestModel", models_directory=models_dir)
    assert result is not None
    assert result.endswith("TestModel.json")

def test_find_model_file_case_insensitive_match(models_dir):
    """Test finding a model file with a case-insensitive fallback."""
    result = loaders.find_model_file("AnotherModel", models_directory=models_dir)
    assert result is not None
    assert result.endswith("anothermodel.json")

def test_find_model_file_not_found(models_dir, caplog):
    """Test that None is returned when a model file is not found."""
    result = loaders.find_model_file("NonExistentModel", models_directory=models_dir)
    assert result is None
    assert "No JSON file found for model 'NonExistentModel'" in caplog.text

def test_find_model_file_directory_not_found(caplog):
    """Test that None is returned when the models directory does not exist."""
    result = loaders.find_model_file("AnyModel", models_directory="non_existent_dir")
    assert result is None
    assert "Models directory 'non_existent_dir' not found" in caplog.text

# ------------------------------------------------------------------------------------------------
# Tests for load_json_file
# ------------------------------------------------------------------------------------------------

def test_load_json_file_success(models_dir):
    """Test successfully loading a valid JSON file."""
    file_path = loaders.find_model_file("TestModel", models_directory=models_dir)
    data = loaders.load_json_file(file_path)
    assert data is not None
    assert data["name"] == "TestModel"

def test_load_json_file_invalid_json(models_dir, caplog):
    """Test loading a file with invalid JSON content."""
    file_path = loaders.find_model_file("InvalidJson", models_directory=models_dir)
    data = loaders.load_json_file(file_path)
    assert data is None
    assert "Invalid JSON format" in caplog.text

def test_load_json_file_not_found(caplog):
    """Test loading a non-existent file."""
    data = loaders.load_json_file("non_existent_file.json")
    assert data is None
    assert "Error loading JSON file" in caplog.text

# ------------------------------------------------------------------------------------------------
# Tests for load_model_data
# ------------------------------------------------------------------------------------------------

@patch('model_scoring.data.loaders.validate_model_data')
def test_load_model_data_success(mock_validate, models_dir, caplog):
    """Test the successful loading and validation of a model."""
    caplog.set_level(logging.INFO) # Set log level to capture INFO messages
    mock_validate.return_value = None  # Simulate successful validation
    data = loaders.load_model_data("TestModel", models_directory=models_dir)
    
    assert data is not None
    assert data["name"] == "TestModel"
    mock_validate.assert_called_once()
    assert "Successfully validated data for model 'TestModel'" in caplog.text

@patch('model_scoring.data.loaders.validate_model_data')
def test_load_model_data_file_not_found(mock_validate, models_dir):
    """Test loading a model whose file does not exist."""
    data = loaders.load_model_data("NonExistentModel", models_directory=models_dir)
    assert data is None
    mock_validate.assert_not_called()

@patch('model_scoring.data.loaders.validate_model_data')
def test_load_model_data_invalid_json(mock_validate, models_dir):
    """Test loading a model with an invalid JSON file."""
    data = loaders.load_model_data("InvalidJson", models_directory=models_dir)
    assert data is None
    mock_validate.assert_not_called()

@patch('model_scoring.data.loaders.validate_model_data')
def test_load_model_data_validation_error(mock_validate, models_dir, caplog):
    """Test that a validation error is handled correctly."""
    mock_validate.side_effect = Exception("Validation failed!")
    data = loaders.load_model_data("TestModel", models_directory=models_dir)
    
    assert data is None
    mock_validate.assert_called_once()
    assert "Error processing model 'TestModel': Validation failed!" in caplog.text 