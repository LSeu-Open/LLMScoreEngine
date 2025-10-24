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
This module provides unit tests for the graph_reporter module.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import os
import pytest
import pandas as pd
from datetime import datetime
from model_scoring.utils.graph_reporter import generate_report

# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------

@pytest.fixture
def test_environment(tmp_path):
    """
    Set up the test environment by creating dummy directories and a CSV report file.
    """
    results_dir = tmp_path / "Results"
    reports_dir = results_dir / "Reports"
    reports_dir.mkdir(parents=True)

    # Create a dummy CSV report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = reports_dir / f'LLM-Scoring-Engine_report_{timestamp}.csv'
    data = {
        'model_name': ['test-model-1', 'test-model-2'],
        'param_count': [10, 80],
        'architecture': ['Transformer', 'MoE'],
        'price': [1.0, 2.0],
        'entity_score': [1, 3], 
        'dev_score': [2, 2], 
        'community_score': [3, 1], 
        'technical_score': [4, 4], 
        'final_score': [2.5, 3.5]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # Create a dummy report_template.html
    template_path = tmp_path / "model_scoring" / "utils" / "report_template.html"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text("<html><body><h1>{{ report_title }}</h1><p>Top model: {{ summary_data.top_model_by_score }}</p></body></html>")

    return tmp_path, data, None

# ------------------------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------------------------

def test_generate_report_success(test_environment, monkeypatch):
    """
    Test the generate_report function for a successful execution.
    """
    print("\n--- Testing generate_report (Success) ---")

    # Arrange: Configure the test environment
    tmp_path, _, _ = test_environment
    monkeypatch.chdir(tmp_path)
    template_dir = tmp_path / "model_scoring" / "utils"

    print("Generating HTML report...")
    # Act: Call the function, passing the test template directory
    generate_report(template_dir=str(template_dir))

    # Assert: Verify the outcome
    reports_dir = tmp_path / "Results" / "Reports"
    report_files = [f for f in os.listdir(reports_dir) if f.endswith('.html')]
    assert len(report_files) == 1, "A single HTML report file should be created."
    
    html_report_path = reports_dir / report_files[0]
    assert html_report_path.name == "model_performance_report.html"

    # Verify the content of the report
    with open(html_report_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    assert "LLM Scoring Engine - Performance Report" in html_content, "The report title should be present in the HTML output."
    assert "Top model: test-model-2 (Score: 3.50)" in html_content, "The top model data should be correctly rendered in the report."

    print("✅ HTML report generated successfully and content verified.") 
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
This module provides unit tests for the graph_reporter module.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import os
import pytest
import pandas as pd
from datetime import datetime
from model_scoring.utils.graph_reporter import generate_report

# ------------------------------------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------------------------------------

@pytest.fixture
def test_environment(tmp_path):
    """
    Set up the test environment by creating dummy directories and a CSV report file.
    """
    results_dir = tmp_path / "Results"
    reports_dir = results_dir / "Reports"
    reports_dir.mkdir(parents=True)

    # Create a dummy CSV report file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = reports_dir / f'LLM-Scoring-Engine_report_{timestamp}.csv'
    data = {
        'model_name': ['test-model-1', 'test-model-2'],
        'param_count': [10, 80],
        'architecture': ['Transformer', 'MoE'],
        'price': [1.0, 2.0],
        'entity_score': [1, 3], 
        'dev_score': [2, 2], 
        'community_score': [3, 1], 
        'technical_score': [4, 4], 
        'final_score': [2.5, 3.5]
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)

    # Create a dummy report_template.html
    template_path = tmp_path / "model_scoring" / "utils" / "report_template.html"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text("<html><body><h1>{{ report_title }}</h1><p>Top model: {{ summary_data.top_model_by_score }}</p></body></html>")

    return tmp_path, data, None

# ------------------------------------------------------------------------------------------------
# Unit Tests
# ------------------------------------------------------------------------------------------------

def test_generate_report_success(test_environment, monkeypatch):
    """
    Test the generate_report function for a successful execution.
    """
    print("\n--- Testing generate_report (Success) ---")

    # Arrange: Configure the test environment
    tmp_path, _, _ = test_environment
    monkeypatch.chdir(tmp_path)
    template_dir = tmp_path / "model_scoring" / "utils"

    print("Generating HTML report...")
    # Act: Call the function, passing the test template directory
    generate_report(template_dir=str(template_dir))

    # Assert: Verify the outcome
    reports_dir = tmp_path / "Results" / "Reports"
    report_files = [f for f in os.listdir(reports_dir) if f.endswith('.html')]
    assert len(report_files) == 1, "A single HTML report file should be created."
    
    html_report_path = reports_dir / report_files[0]
    assert html_report_path.name == "model_performance_report.html"

    # Verify the content of the report
    with open(html_report_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    assert "LLM Scoring Engine - Performance Report" in html_content, "The report title should be present in the HTML output."
    assert "Top model: test-model-2 (Score: 3.50)" in html_content, "The top model data should be correctly rendered in the report."

    print("✅ HTML report generated successfully and content verified.") 
>>>>>>> theirs
