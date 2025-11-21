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
from typing import Any, Dict, Optional, Sequence
from model_scoring.utils.logging import configure_console_only_logging
from model_scoring.utils.csv_reporter import generate_csv_report
from model_scoring.utils.graph_reporter import (
    generate_report as generate_graph_report,
)
from llmscore.actions.base import ActionExecutionError
from llmscore.actions.catalog import register_default_actions
from llmscore.actions.registry import ActionRegistry

# ------------------------------------------------------------------------------------------------
# CLI Setup
# ------------------------------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Score LLM models based on various benchmarks and criteria."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "models",
        nargs="*",
        help=(
            "Names of models to score (JSON filename without extension). "
            "Ignored when --all is provided."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Score every model JSON located in the Models directory.",
    )
    parser.add_argument(
        "--models-dir",
        default="Models",
        help="Directory containing model JSON files (defaults to ./Models).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logs and print only final scores.",
    )
    parser.add_argument(
        "--config",
        help="Optional scoring configuration file (YAML or JSON).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Generate a CSV report from the existing scoring results.",
    )
    parser.add_argument(
        "--graph",
        action="store_true",
        help=(
            "Generate an HTML graph report from the existing scoring "
            "results."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s Beta v0.5",
    )
    return parser.parse_args(argv)

# ------------------------------------------------------------------------------------------------
# Main script
# ------------------------------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Legacy scoring entry point."""

    args = parse_args(argv)
    try:
        if args.csv:
            generate_csv_report()
            print("[*] CSV report generated successfully.")
            return

        if args.graph:
            print("[*] Generating HTML graph report...")
            generate_graph_report()
            print("[*] HTML graph report generated successfully.")
            return

        configure_console_only_logging(quiet=args.quiet)

        if args.all:
            models_dir = args.models_dir
            if not os.path.isdir(models_dir):
                print(
                    f"[-] Models directory not found: {models_dir}. "
                    "Provide a valid path via --models-dir."
                )
                return
            model_names = [
                path.removesuffix(".json")
                for path in os.listdir(models_dir)
                if path.endswith(".json")
            ]
        else:
            model_names = list(args.models or [])

        if not model_names:
            print(
                "[-] No models specified. Provide at least one model name or "
                "use the --all flag."
            )
            return

        registry = ActionRegistry()
        register_default_actions(registry)
        inputs: Dict[str, Any] = {
            "models": model_names,
            "quiet": args.quiet,
            "results_dir": "Results",
            "models_dir": args.models_dir,
        }
        if args.config:
            print(f"[*] Loading custom configuration from: {args.config}")
            inputs["config_path"] = args.config
        result = registry.run("score.batch", inputs=inputs)
        payload = result.output or {}
        successes = payload.get("successes", 0)
        failures = payload.get("failures", 0)
        if args.quiet and "results" in payload:
            for item in payload["results"]:
                if item.get("status") == "success":
                    scores = item.get("scores", {})
                    final_score = scores.get("final_score")
                    if final_score is not None:
                        print(f"{item['model']}: {final_score:.4f}")
        else:
            print("=" * 60)
            print(
                f"[+] Batch processing completed successfully for "
                f"{successes} models"
            )
            if failures:
                print(f"[!] {failures} models failed.")
            print("=" * 60)

    except (ActionExecutionError, Exception) as e:
        print(f"\n[-] Processing failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
