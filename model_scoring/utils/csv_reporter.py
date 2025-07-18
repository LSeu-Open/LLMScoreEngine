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
    reports_dir = os.path.join(results_dir, 'Reports')
    os.makedirs(reports_dir, exist_ok=True)

    models_dir = 'Models'
    project_name = "LLM-Scoring-Engine"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(reports_dir, f'{project_name}_report_{timestamp}.csv')
    headers = [
        'model_name',
        'param_count',
        'architecture',
        'input_price',
        'output_price',
        'entity_score', 
        'dev_score', 
        'community_score', 
        'technical_score', 
        'final_score'
    ]

    json_files = [f for f in os.listdir(results_dir) if f.endswith('_results.json')]

    if not json_files:
        print("No JSON result files found in the Results directory.")
        return

    try:
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()
    
            for json_file in json_files:
                results_path = os.path.join(results_dir, json_file)
                model_info_file = json_file.replace('_results.json', '.json')
                model_info_path = os.path.join(models_dir, model_info_file)
    
                try:
                    with open(results_path, 'r') as f:
                        results_data = json.load(f)
    
                    model_data = {}
                    if os.path.exists(model_info_path):
                        with open(model_info_path, 'r') as f:
                            model_data = json.load(f)
                    else:
                        print(f"Warning: Model file not found for {json_file}")
    
                    scores = results_data.get('scores', {})
                    model_specs = model_data.get('model_specs', {})
                    row = {
                        'model_name': results_data.get('model_name'),
                        'param_count': model_specs.get('param_count'),
                        'architecture': model_specs.get('architecture'),
                        'input_price': model_specs.get('input_price'),
                        'output_price': model_specs.get('output_price'),
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
        print(f"CSV report successfully generated and saved to {output_file}")
    except Exception as e:
        print(f"Error writing CSV report to {output_file}: {e}") 