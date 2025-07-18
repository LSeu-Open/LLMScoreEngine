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
Tests for the main scoring runner.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import pytest
from unittest.mock import patch, MagicMock, call

from model_scoring import run_scoring

# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------

@pytest.fixture
def mock_model_data():
    """Provides a mock of loaded model data."""
    return {
        "entity_benchmarks": {"benchmark1": 0.8},
        "dev_benchmarks": {"benchmark2": 0.9},
        "community_score": {"lm_sys_arena_score": 1200, "hf_score": 9.5},
        "model_specs": {"price": 1.0, "context_window": 8192, "param_count": 7, "architecture": "transformer"}
    }

@pytest.fixture
def mock_scoring_results():
    """Provides a mock of the results returned by the scoring process."""
    return {
        'model_name': 'TestModel',
        'scores': {
            'entity_score': 85.0, 'dev_score': 90.0, 'community_score': 92.0,
            'technical_score': 88.0, 'final_score': 88.75
        },
        'input_data': {}
    }

# ------------------------------------------------------------------------------------------------
# Tests for run_scoring
# ------------------------------------------------------------------------------------------------

@patch('model_scoring.run_scoring.load_model_data')
@patch('model_scoring.run_scoring.ModelScorer')
def test_run_scoring_success(mock_model_scorer, mock_load_data, mock_model_data):
    """Test the run_scoring function for a successful execution."""
    print("\n--- Testing run_scoring (Success) ---")
    
    # Arrange: Configure mocks
    mock_load_data.return_value = mock_model_data
    
    mock_scorer_instance = MagicMock()
    mock_scorer_instance.calculate_final_score.return_value = 95.0
    mock_scorer_instance.entity_score = 85.0
    mock_scorer_instance.dev_score = 90.0
    mock_scorer_instance.community_score = 92.0
    mock_scorer_instance.technical_score = 88.0
    mock_model_scorer.return_value = mock_scorer_instance

    print("Running scoring for 'TestModel'...")
    # Act: Call the function
    results = run_scoring.run_scoring('TestModel', quiet=False, scoring_config=None)

    # Assert: Verify outcomes
    mock_load_data.assert_called_once_with('TestModel', run_scoring.MODELS_DIR, scoring_config=None)
    mock_model_scorer.assert_called_once_with('TestModel', scoring_config=None)
    mock_scorer_instance.calculate_final_score.assert_called_once()
    
    assert results is not None
    assert results['model_name'] == 'TestModel'
    assert results['scores']['final_score'] == 95.0
    print("✅ Scoring successful and results are structured correctly.")

@patch('model_scoring.run_scoring.load_model_data')
def test_run_scoring_data_load_failure(mock_load_data, caplog):
    """Test run_scoring when model data loading fails."""
    print("\n--- Testing run_scoring (Data Load Failure) ---")
    
    # Arrange: Simulate data loading failure
    mock_load_data.return_value = None
    
    print("Simulating failure to load data for 'TestModel'...")
    # Act: Call the function
    results = run_scoring.run_scoring('TestModel', scoring_config=None)

    # Assert: Verify it returns None and logs an error
    assert results is None
    assert "Failed to load data for model 'TestModel'" in caplog.text
    print("✅ Correctly handled data load failure.")

# ------------------------------------------------------------------------------------------------
# Tests for batch_process_models
# ------------------------------------------------------------------------------------------------

@patch('model_scoring.run_scoring.run_scoring')
@patch('os.makedirs')
@patch('builtins.open', new_callable=MagicMock)
@patch('json.dump')
def test_batch_process_models_success(mock_json_dump, mock_open, mock_makedirs, mock_run_scoring, mock_scoring_results, tmp_path):
    """Test batch processing for multiple models with successful outcomes."""
    print("\n--- Testing batch_process_models (Success) ---")
    
    # Arrange: Configure mocks
    model_list = ['ModelA', 'ModelB']
    results_dir = str(tmp_path / "results")
    mock_run_scoring.return_value = mock_scoring_results

    print(f"Processing batch for models: {model_list}")
    # Act: Call the batch processing function
    run_scoring.batch_process_models(model_list, results_directory=results_dir, quiet=False, scoring_config=None)

    # Assert: Verify calls
    mock_makedirs.assert_called_once_with(results_dir, exist_ok=True)
    assert mock_run_scoring.call_count == 2
    mock_run_scoring.assert_has_calls([
        call('ModelA', run_scoring.MODELS_DIR, quiet=False, scoring_config=None), 
        call('ModelB', run_scoring.MODELS_DIR, quiet=False, scoring_config=None)
    ])
    
    assert mock_open.call_count == 2 
    mock_json_dump.assert_called()
    print("✅ Batch processing successful, results saved for all models.")

@patch('model_scoring.run_scoring.run_scoring')
@patch('os.makedirs')
@patch('builtins.open', new_callable=MagicMock)
@patch('json.dump')
def test_batch_process_with_failure(mock_json_dump, mock_open, mock_makedirs, mock_run_scoring, caplog, tmp_path, mock_scoring_results):
    """Test batch processing where one model fails to score."""
    print("\n--- Testing batch_process_models (One Failure) ---")
    
    # Arrange: Configure mocks to simulate one failure
    model_list = ['SuccessModel', 'FailModel']
    results_dir = str(tmp_path / "results")
    
    # Side effect allows different return values for consecutive calls
    mock_run_scoring.side_effect = [mock_scoring_results, None]

    print(f"Processing batch for {model_list}, where 'FailModel' will fail...")
    # Act: Call the batch processing function
    with caplog.at_level('INFO'):
        run_scoring.batch_process_models(model_list, results_directory=results_dir, quiet=False, scoring_config=None)

    # Assert: Verify outcomes
    assert mock_run_scoring.call_count == 2
    assert "Results successfully saved" in caplog.text
    assert "Failed to generate results for FailModel" in caplog.text
    assert mock_open.called  # Verify that open was called for the successful model
    assert mock_json_dump.called # Verify that dump was called
    print("✅ Batch processing handled a failed model correctly and continued.") 
