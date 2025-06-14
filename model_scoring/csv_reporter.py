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
This module provides functionality to generate a CSV report from the JSON results in the Results directory.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import csv
import json
import os
from datetime import datetime

# ------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------

def generate_csv_report():
    """
    Generates a CSV report from the JSON results in the Results directory.
    """
    results_dir = 'Results'
    project_name = "LLM-Scoring-Engine"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f'{project_name}_report_{timestamp}.csv')
    headers = [
        'model_name', 
        'entity_score', 
        'dev_score', 
        'community_score', 
        'technical_score', 
        'final_score'
    ]

    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    if not json_files:
        print("No JSON result files found in the Results directory.")
        return

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        writer.writeheader()

        for json_file in json_files:
            file_path = os.path.join(results_dir, json_file)
            with open(file_path, 'r') as f:
                try:
                    data = json.load(f)
                    scores = data.get('scores', {})
                    row = {
                        'model_name': data.get('model_name'),
                        'entity_score': scores.get('entity_score'),
                        'dev_score': scores.get('dev_score'),
                        'community_score': scores.get('community_score'),
                        'technical_score': scores.get('technical_score'),
                        'final_score': scores.get('final_score')
                    }
                    writer.writerow(row)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from {json_file}")
                except Exception as e:
                    print(f"Warning: Could not process {json_file}. Error: {e}") 