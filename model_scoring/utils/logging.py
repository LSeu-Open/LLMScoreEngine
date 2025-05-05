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
Logging configuration for the model scoring system.

This module provides utilities for setting up logging with standard formatting
and handlers for both file and console output.
"""

import logging
from ..core.constants import LOG_FILE

def configure_logging(level=logging.INFO, log_file=LOG_FILE, console_output=True):
    """
    Configure logging with both file and console output.
    
    Args:
        level (int): Logging level (default: logging.INFO)
        log_file (str): Path to log file (default: from LOG_FILE constant)
        console_output (bool): Whether to include console output (default: True)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Reset any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Create handlers
    handlers = []
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    
    # Console handler (optional)
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # Configure basic logging
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True  # Force override any existing configuration
    )
    
    # Get and return logger
    logger = logging.getLogger(__name__)
    return logger

def configure_console_only_logging(level=logging.INFO):
    """
    Configure logging with only console output and simplified formatting.
    
    Useful for CLI applications where only immediate terminal output is needed.
    
    Args:
        level (int): Logging level (default: logging.INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Reset any existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    # Configure basic logging with simplified format
    logging.basicConfig(
        level=level,
        format='%(message)s',
        force=True  # Force override any existing configuration
    )
    
    # Get and return logger
    logger = logging.getLogger(__name__)
    return logger 