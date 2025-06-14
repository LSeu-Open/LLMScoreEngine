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
This module provides utilities for dynamically loading configuration files.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

def load_config_from_path(config_path: str) -> ModuleType:
    """
    Dynamically load a Python configuration file from a given path.

    Args:
        config_path (str): The absolute or relative path to the Python config file.

    Returns:
        ModuleType: The loaded configuration module.
        
    Raises:
        FileNotFoundError: If the configuration file does not exist.
        Exception: If there is an error loading the module.
    """
    config_file = Path(config_path).resolve()

    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {config_file}")

    # Use the file name as the module name
    module_name = config_file.stem

    # Create a module spec from the file path
    spec = importlib.util.spec_from_file_location(module_name, config_file)
    if spec is None or spec.loader is None:
        raise Exception(f"Could not create module spec for {config_file}")

    # Create a new module based on the spec
    config_module = importlib.util.module_from_spec(spec)

    # Add the module to sys.modules to handle relative imports within the config
    sys.modules[module_name] = config_module

    try:
        # Execute the module to load its contents
        spec.loader.exec_module(config_module)
    except Exception as e:
        # Remove the module from sys.modules if it fails to load
        del sys.modules[module_name]
        raise Exception(f"Failed to load configuration from {config_file}: {e}") from e

    return config_module 