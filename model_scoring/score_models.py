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

# This script is the main entry point for running the model scoring system.
# It provides backward compatibility with the previous version.

# This is the Beta v0.5 of the scoring system

# ------------------------------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------------------------------

import argparse
from typing import List
from model_scoring.run_scoring import batch_process_models
from model_scoring.utils.logging import configure_console_only_logging

# ------------------------------------------------------------------------------------------------
# CLI Setup
# ------------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Score LLM models based on various benchmarks and criteria.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "models",
        nargs="*",
        help="Names of models to score. If not provided, defaults to example models."
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s Beta v1.0"
    )
    
    return parser.parse_args()

# ------------------------------------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------------------------------------

def main(models: List[str] = None) -> None:
    """
    Main function to run the scoring system.
    
    Args:
        models: List of model names to process. If None, uses default models.
    """
    try:
        # Configure logging
        configure_console_only_logging()
        
        # Use provided models
        model_names = models or []
        if not model_names:
            print("\n[-] No models specified. Please provide at least one model name")
            return

        # Run batch processing
        batch_process_models(model_names)
            
    except Exception as e:
        print(f"\n[-] Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    args = parse_args()
    main(args.models)