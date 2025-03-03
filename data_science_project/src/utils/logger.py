import logging
import logging.config
from pathlib import Path
from typing import Optional

from config import LOG_CONFIG, PROJECT_ROOT

def setup_logger(
    name: str,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        log_file: Optional log file path. If not provided, uses default from config
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Update log file path in config if provided
    if log_file:
        LOG_CONFIG["handlers"]["file"]["filename"] = str(log_dir / log_file)
    else:
        LOG_CONFIG["handlers"]["file"]["filename"] = str(log_dir / "pipeline.log")
    
    # Apply configuration
    logging.config.dictConfig(LOG_CONFIG)
    logger = logging.getLogger(name)
    
    return logger

def log_decorator(func):
    """
    Decorator to log function entry and exit with parameters.
    
    Args:
        func: Function to decorate
        
    Returns:
        wrapper: Decorated function
    """
    logger = logging.getLogger(func.__module__)
    
    def wrapper(*args, **kwargs):
        logger.debug(
            f"Entering {func.__name__} with args: {args}, kwargs: {kwargs}"
        )
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__}")
            return result
        except Exception as e:
            logger.error(
                f"Exception in {func.__name__}: {str(e)}",
                exc_info=True
            )
            raise
    
    return wrapper