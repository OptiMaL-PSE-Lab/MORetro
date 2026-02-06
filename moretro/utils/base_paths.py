import os
import sys
from pathlib import Path

# Root directory of project
ROOT_DIR = Path(__file__).parent.parent.parent
# Package directory
PACKAGE_DIR = ROOT_DIR / "moretro"
# Config directory
CONFIG_DIR = PACKAGE_DIR / "config"

# Executable directory
EXECUTABLE_DIR = Path.cwd()

# Define default model and log dirs (can be overwritten by env variables)
DEFAULT_MODEL_DIR = EXECUTABLE_DIR / "models"
DEFAULT_LOG_DIR = EXECUTABLE_DIR / "logs"

# Allow overriding model and log dirs via environment variables
MODELS_DIR = Path(os.getenv("MORETRO_MODEL_DIR", DEFAULT_MODEL_DIR))
LOG_DIR = Path(os.getenv("MORETRO_LOG_DIR", DEFAULT_LOG_DIR))

# Ensure model directory exists, if not, raise error
if not MODELS_DIR.exists() and "pytest" not in sys.modules:
    raise FileNotFoundError(
        f"Model directory {MODELS_DIR} does not exist. Please first download required models from FigShare."
    )
