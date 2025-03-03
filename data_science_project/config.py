from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Data processing settings
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Model parameters
MODEL_PARAMS = {
    "random_state": RANDOM_SEED,
    "max_iter": 1000,
    # TODO: Add more model parameters as needed
}

# Feature preprocessing
NUMERICAL_FEATURES = [
    # TODO: Add your numerical feature names
]

CATEGORICAL_FEATURES = [
    # TODO: Add your categorical feature names
]

TARGET_COLUMN = "target"  # TODO: Replace with your target column name

# Logging configuration
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "INFO",
            "formatter": "standard",
            "filename": "pipeline.log",
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}