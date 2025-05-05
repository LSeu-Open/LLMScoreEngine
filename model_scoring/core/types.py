"""
Type definitions for the model scoring system.

This module defines type aliases used throughout the codebase to improve
readability and maintainability.
"""

from typing import Dict, Any, Optional, Union, List

# Try to import TypedDict from typing, or fall back to typing_extensions for Python < 3.8
try:
    from typing import TypedDict
except ImportError:
    try:
        from typing_extensions import TypedDict
    except ImportError:
        # Define a simple replacement if TypedDict is not available
        class TypedDict(dict):
            pass

# Type aliases
ModelData = Dict[str, Any]  # Raw model data containing all attributes and scores
BenchmarkScores = Dict[str, Optional[float]]  # Benchmark results, allowing for missing scores
ModelSpecs = Dict[str, float]  # Technical specifications of the model (price, context window, etc)
ScoringResults = Dict[str, Union[str, Dict[str, float], ModelData]]  # Final scoring output format

# TypedDict definitions for more precise typing
class ModelSpecifications(TypedDict):
    price: float
    context_window: int
    param_count: float 