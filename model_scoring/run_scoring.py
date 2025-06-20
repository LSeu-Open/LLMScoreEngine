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
Main module for the model scoring system.

This module provides the main functionality for scoring models, including
single-model scoring and batch processing.
"""

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import os
import json
import time
import logging
from typing import List, Optional
import sys
from pathlib import Path
from types import ModuleType
from .core.constants import MODELS_DIR, RESULTS_DIR
from .core.types import ScoringResults
from .data.loaders import load_model_data
from .scoring.models_scoring import ModelScorer
from .utils.logging import configure_console_only_logging

# Add project root to sys.path for absolute imports.
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------------------
# Functions
# ------------------------------------------------------------------------------------------------

def run_scoring(model_name: str, models_directory: str = MODELS_DIR, 
                quiet: bool = False, scoring_config: Optional[ModuleType] = None) -> Optional[ScoringResults]:
    """
    Run the scoring process for a given model by loading its JSON from the Models directory.
    
    Args:
        model_name: Name of the model to score
        models_directory: Path to the directory containing model JSONs
        quiet: If True, suppresses detailed score breakdown output.
        scoring_config: A loaded configuration module.
    
    Returns:
        Scoring results if successful, None otherwise
    """
    logger.info(f"Starting scoring process for model '{model_name}'")
    
    # Load and validate the model data
    data = load_model_data(model_name, models_directory, scoring_config=scoring_config)
    if not data:
        logger.error(f"Failed to load data for model '{model_name}'")
        return None

    try:
        # Initialize scorer with model name and config
        scorer = ModelScorer(model_name, scoring_config=scoring_config)
        
        # Extract data sections from the loaded JSON
        entity_benchmarks = data.get('entity_benchmarks', {})
        dev_benchmarks = data.get('dev_benchmarks', {})
        community_scores_data = data.get('community_score', {})
        model_specs = data.get('model_specs', {})

        # Group inputs for the main scoring function
        community_inputs = {
            'lm_sys_arena_elo_rating': community_scores_data.get('lm_sys_arena_score'),
            'hf_score': community_scores_data.get('hf_score')
        }
        
        tech_inputs = {
            'price': model_specs.get('price'),
            'context_window': model_specs.get('context_window'),
            'param_count': model_specs.get('param_count'),
            'architecture': model_specs.get('architecture')
        }

        # Run the entire scoring process with a single method call
        final_score = scorer.calculate_final_score(
            entity_benchmarks=entity_benchmarks,
            dev_benchmarks=dev_benchmarks,
            community_inputs=community_inputs,
            tech_inputs=tech_inputs,
            quiet=quiet
        )

        # Prepare results dictionary from the scorer instance attributes
        results: ScoringResults = {
            'model_name': model_name,
            'scores': {
                'entity_score': scorer.entity_score,
                'dev_score': scorer.dev_score,
                'community_score': scorer.community_score,
                'technical_score': scorer.technical_score,
                'final_score': final_score,
            },
            'input_data': data
        }
        
        logger.info(f"Successfully completed scoring for model '{model_name}'")
        return results

    except Exception as e:
        logger.error(f"Error during scoring process: {str(e)}")
        return None

def _display_final_score(results: ScoringResults) -> None:
    """
    Display the final score in a clean, quiet format.

    Args:
        results: The scoring results for a model.
    """
    model_name = results.get('model_name', 'Unknown Model')
    final_score = results.get('scores', {}).get('final_score', 'N/A')
    print(f"{model_name}: {final_score:.4f}")

def batch_process_models(model_names: List[str], models_directory: str = MODELS_DIR, 
                         results_directory: str = RESULTS_DIR, quiet: bool = False,
                         scoring_config: Optional[ModuleType] = None) -> None:
    """
    Process multiple models in batch mode.
    
    Args:
        model_names: List of model names to process
        models_directory: Directory containing model JSON files
        results_directory: Directory to save results to
        quiet: Whether to display only final scores
        scoring_config: A loaded configuration module.
    """
    start_time = time.time()
    
    # Create Results directory if it doesn't exist
    os.makedirs(results_directory, exist_ok=True)
    
    total_models = len(model_names)
    all_results = []
    
    # Different header based on number of models
    if total_models == 1:
        logger.info("\n[*] Processing Single Model")
    else:
        logger.info(f"\n[*] Batch Processing {total_models} Models")
        
    # Process each model sequentially
    for index, model_name in enumerate(model_names, 1):
        if total_models > 1:
            logger.info("\n" + "=" * 60)
            logger.info(f"Model {index}/{total_models}: {model_name}")
            logger.info("=" * 60)
        else:
            logger.info("\n" + "=" * 60)
        
        logger.info("[>] Starting evaluation...\n")
        
        # Run scoring pipeline for current model
        results = run_scoring(model_name, models_directory, quiet=quiet, scoring_config=scoring_config)
        
        if results:
            all_results.append(results)
            output_file = os.path.join(results_directory, f"{model_name}_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info("[+] Results successfully saved to:")
            logger.info(f"    {output_file}\n")
        else:
            logger.error(f"[-] Failed to generate results for {model_name}\n")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    if quiet:
        for res in all_results:
            _display_final_score(res)
        return
        
    # Final summary
    logger.info("=" * 60)
    if total_models == 1:
        logger.info("[+] Processing completed successfully")
    else:
        logger.info(f"[+] Batch processing completed successfully for all {total_models} models")
    logger.info(f"[*] Total processing time: {elapsed_time:.2f} seconds")
    logger.info("=" * 60 + "\n")

# ------------------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------------------

def main():
    """Main entry point for the scoring system."""
    # Configure logging
    configure_console_only_logging()
    
    # List of model names to process - can be expanded as needed
    model_names = ["Command A"]
    
    # Run batch processing
    batch_process_models(model_names)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"\n[-] Processing failed: {str(e)}") 
