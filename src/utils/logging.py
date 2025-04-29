import logging
import os

def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Sets up a logger that prints to console and optionally writes to a file.

    Args:
        log_file (str, optional): Path to log file. If None, no file logging.
        log_level (int, optional): Logging level (e.g., logging.INFO, logging.DEBUG)

    Returns:
        logging.Logger: Configured logger object
    """
    logger = logging.getLogger("SafeRideDetectionLogger")
    logger.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
