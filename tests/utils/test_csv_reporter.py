<<<<<<< ours
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
This module provides unit tests for the csv_reporter module.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import os
import json
import csv
import pytest
from model_scoring.utils.csv_reporter import generate_csv_report

# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------

@pytest.fixture
def test_environment(tmp_path):
    """
    Set up the test environment by creating dummy directories and files.
    """
    results_dir = tmp_path / "Results"
    models_dir = tmp_path / "Models"
    reports_dir = results_dir / "Reports"

    results_dir.mkdir()
    models_dir.mkdir()
    reports_dir.mkdir()

    model1_results = {
        "model_name": "test-model-1",
        "scores": {
            "entity_score": 1, "dev_score": 2, "community_score": 3, "technical_score": 4, "final_score": 2.5
        }
    }
    model1_info = {
        "model_specs": {
            "param_count": "10B", "architecture": "Transformer", "price": "Free"
        }
    }

    with open(results_dir / 'test-model-1_results.json', 'w') as f:
        json.dump(model1_results, f)
    with open(models_dir / 'test-model-1.json', 'w') as f:
        json.dump(model1_info, f)

    return tmp_path, model1_results, model1_info

# ------------------------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------------------------

def test_generate_csv_report(test_environment, monkeypatch):
    """
    Test the generate_csv_report function for a successful execution.
    """
    print("\n--- Testing generate_csv_report (Success) ---")

    # Arrange: Configure the test environment
    tmp_path, model1_results, model1_info = test_environment
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / "Results" / "Reports"

    print("Generating CSV report...")
    # Act: Call the function
    generate_csv_report()

    # Assert: Verify the outcome
    report_files = [f for f in os.listdir(reports_dir) if f.endswith('.csv')]
    assert len(report_files) == 1, "A single report file should be created."

    report_file_path = reports_dir / report_files[0]
    with open(report_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        assert len(rows) == 1, "The report should contain exactly one data row."
        
        row = rows[0]
        assert row['model_name'] == model1_results['model_name']
        assert row['param_count'] == model1_info['model_specs']['param_count']
        assert row['architecture'] == model1_info['model_specs']['architecture']
        assert row['price'] == model1_info['model_specs']['price']
        assert float(row['entity_score']) == model1_results['scores']['entity_score']
        assert float(row['dev_score']) == model1_results['scores']['dev_score']
        assert float(row['community_score']) == model1_results['scores']['community_score']
        assert float(row['technical_score']) == model1_results['scores']['technical_score']
        assert float(row['final_score']) == model1_results['scores']['final_score']

    print("✅ CSV report generated successfully with correct data.") 
|||||||
=======
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
This module provides unit tests for the csv_reporter module.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import os
import json
import csv
import pytest
from model_scoring.utils.csv_reporter import generate_csv_report

# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------

@pytest.fixture
def test_environment(tmp_path):
    """
    Set up the test environment by creating dummy directories and files.
    """
    results_dir = tmp_path / "Results"
    models_dir = tmp_path / "Models"
    reports_dir = results_dir / "Reports"

    results_dir.mkdir()
    models_dir.mkdir()
    reports_dir.mkdir()

    model1_results = {
        "model_name": "test-model-1",
        "scores": {
            "entity_score": 1, "dev_score": 2, "community_score": 3, "technical_score": 4, "final_score": 2.5
        }
    }
    model1_info = {
        "model_specs": {
            "param_count": "10B", "architecture": "Transformer", "price": "Free"
        }
    }

    with open(results_dir / 'test-model-1_results.json', 'w') as f:
        json.dump(model1_results, f)
    with open(models_dir / 'test-model-1.json', 'w') as f:
        json.dump(model1_info, f)

    return tmp_path, model1_results, model1_info

# ------------------------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------------------------

def test_generate_csv_report(test_environment, monkeypatch):
    """
    Test the generate_csv_report function for a successful execution.
    """
    print("\n--- Testing generate_csv_report (Success) ---")

    # Arrange: Configure the test environment
    tmp_path, model1_results, model1_info = test_environment
    monkeypatch.chdir(tmp_path)
    reports_dir = tmp_path / "Results" / "Reports"

    print("Generating CSV report...")
    # Act: Call the function
    generate_csv_report()

    # Assert: Verify the outcome
    report_files = [f for f in os.listdir(reports_dir) if f.endswith('.csv')]
    assert len(report_files) == 1, "A single report file should be created."

    report_file_path = reports_dir / report_files[0]
    with open(report_file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
        assert len(rows) == 1, "The report should contain exactly one data row."
        
        row = rows[0]
        assert row['model_name'] == model1_results['model_name']
        assert row['param_count'] == model1_info['model_specs']['param_count']
        assert row['architecture'] == model1_info['model_specs']['architecture']
        assert row['price'] == model1_info['model_specs']['price']
        assert float(row['entity_score']) == model1_results['scores']['entity_score']
        assert float(row['dev_score']) == model1_results['scores']['dev_score']
        assert float(row['community_score']) == model1_results['scores']['community_score']
        assert float(row['technical_score']) == model1_results['scores']['technical_score']
        assert float(row['final_score']) == model1_results['scores']['final_score']

    print("✅ CSV report generated successfully with correct data.") 
>>>>>>> theirs
