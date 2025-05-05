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

import os
import json
import time
import logging
from typing import List, Optional

from .core.constants import MODELS_DIR, RESULTS_DIR
from .core.types import ScoringResults, ModelData
from .data.loaders import load_model_data
from .scoring.models_scoring import ModelScorer
from .utils.logging import configure_console_only_logging

logger = logging.getLogger(__name__)

def run_scoring(model_name: str, models_directory: str = MODELS_DIR) -> Optional[ScoringResults]:
    """
    Run the scoring process for a given model by loading its JSON from the Models directory.
    
    Args:
        model_name: Name of the model to score
        models_directory: Path to the directory containing model JSONs
    
    Returns:
        Scoring results if successful, None otherwise
    """
    logger.info(f"Starting scoring process for model '{model_name}'")
    
    # Load and validate the model data
    data = load_model_data(model_name, models_directory)
    if not data:
        logger.error(f"Failed to load data for model '{model_name}'")
        return None

    try:
        # Initialize scorer with model name
        scorer = ModelScorer(model_name)
        
        # Extract data from JSON
        entity_benchmarks = data.get('entity_benchmarks', {})
        dev_benchmarks = data.get('dev_benchmarks', {})
        community_score = data.get('community_score', 0)
        model_specs = data.get('model_specs', {})

        # Calculate average benchmark performance
        available_scores = (
            [score for score in entity_benchmarks.values() if score is not None] +
            [score for score in dev_benchmarks.values() if score is not None]
        )
        avg_performance = sum(available_scores) / len(available_scores) * 100  # Convert to percentage

        # Calculate size/performance ratio
        size_perf_ratio = scorer.calculate_size_perf_ratio(avg_performance, model_specs['param_count'])

        # Calculate scores
        entity_score = scorer.calculate_entity_benchmarks(entity_benchmarks)
        dev_score = scorer.calculate_dev_benchmarks(dev_benchmarks)
        external_score = scorer.calculate_external_benchmarks(entity_benchmarks, dev_benchmarks)
        community_score = scorer.calculate_community_score(community_score)
        technical_score = scorer.calculate_technical_score(
            price=model_specs['price'],
            context_window=model_specs['context_window'],
            size_perf_ratio=size_perf_ratio
        )

        # Set scores
        scorer.external_score = external_score
        scorer.community_score = community_score 
        scorer.technical_score = technical_score

        # Calculate final score
        final_score = scorer.calculate_final_score()

        # Prepare results dictionary
        results: ScoringResults = {
            'model_name': model_name,
            'scores': {
                'entity_score': entity_score,
                'dev_score': dev_score,
                'external_score': external_score,
                'community_score': community_score,
                'technical_score': technical_score,
                'final_score': final_score,
                'avg_performance': avg_performance,
                'size_perf_ratio': size_perf_ratio
            },
            'input_data': data  # Include input data for reference
        }
        
        logger.info(f"Successfully completed scoring for model '{model_name}'")
        return results

    except Exception as e:
        logger.error(f"Error during scoring process: {str(e)}")
        return None

def batch_process_models(model_names: List[str], models_directory: str = MODELS_DIR, 
                         results_directory: str = RESULTS_DIR) -> None:
    """
    Process multiple models in batch mode.
    
    Args:
        model_names: List of model names to process
        models_directory: Directory containing model JSON files
        results_directory: Directory to save results to
    """
    start_time = time.time()
    
    # Create Results directory if it doesn't exist
    os.makedirs(results_directory, exist_ok=True)
    
    total_models = len(model_names)
    
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
        results = run_scoring(model_name, models_directory)
        
        if results:
            output_file = os.path.join(results_directory, f"{model_name}_results.json")
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=4)
            logger.info("[+] Results successfully saved to:")
            logger.info(f"    {output_file}\n")
        else:
            logger.error(f"[-] Failed to generate results for {model_name}\n")
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Final summary
    logger.info("=" * 60)
    if total_models == 1:
        logger.info("[+] Processing completed successfully")
    else:
        logger.info(f"[+] Batch processing completed successfully for all {total_models} models")
    logger.info(f"[*] Total processing time: {elapsed_time:.2f} seconds")
    logger.info("=" * 60 + "\n")

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