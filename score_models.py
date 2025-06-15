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
import os
from typing import List
from model_scoring.run_scoring import batch_process_models
from model_scoring.utils.logging import configure_console_only_logging
from model_scoring.utils.config_loader import load_config_from_path
from model_scoring.csv_reporter import generate_csv_report
from config import scoring_config as default_scoring_config

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
        '--all',
        action='store_true',
        help='Score all models in the Models folder.'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress all informational output and only show the final scores.'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to a custom scoring configuration file.'
    )
    
    parser.add_argument(
        '--csv',
        action='store_true',
        help='Generate a CSV report from the results.'
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s Beta v0.5"
    )
    
    return parser.parse_args()

# ------------------------------------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------------------------------------

def main() -> None:
    """
    Main function to run the scoring system.
    
    Args:
        models: List of model names to process. If None, uses default models.
    """
    args = parse_args()
    try:
        if args.csv:
            generate_csv_report()
            print("[*] CSV report generated successfully.")
            return

        # Load scoring configuration
        if args.config:
            print(f"[*] Loading custom configuration from: {args.config}")
            scoring_config = load_config_from_path(args.config)
        else:
            scoring_config = default_scoring_config

        # Configure logging
        configure_console_only_logging(quiet=args.quiet)

        model_names = []
        if args.all:
            model_names = [f.removesuffix('.json') for f in os.listdir('Models') if f.endswith('.json')]
        elif args.models:
            model_names = args.models
        
        # Use provided models
        if not model_names:
            print("\n[-] No models specified. Please provide at least one model name or use the --all flag.")
            return

        # Run batch processing
        batch_process_models(model_names, quiet=args.quiet, scoring_config=scoring_config)
            
    except Exception as e:
        print(f"\n[-] Processing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
