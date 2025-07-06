import yaml
import os
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

MODEL_SETTINGS_FILENAME = "model_settings.yaml"

def load_model_settings(config_dir: Optional[str] = None) -> Dict:
    """
    Loads model settings from a YAML file.

    Args:
        config_dir (str, optional): The directory where 'model_settings.yaml' is located.
                                    If None, defaults to the current working directory,
                                    then tries one level up (common for running scripts in subdirs).

    Returns:
        Dict: A dictionary containing the model settings, or an empty dict if the file is not found or is invalid.
    """
    search_paths = []
    if config_dir:
        search_paths.append(os.path.join(config_dir, MODEL_SETTINGS_FILENAME))
    else:
        # Default search paths: current dir, then parent dir
        # This helps when running scripts from project root or from a subdirectory like 'cli'
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")) # Assumes this file is in tradingagents/config/
        search_paths.extend([
            os.path.join(os.getcwd(), MODEL_SETTINGS_FILENAME),
            os.path.join(project_root, MODEL_SETTINGS_FILENAME), # Check project root
            MODEL_SETTINGS_FILENAME # Relative to current execution path
        ])

    filepath_found = None
    for path_to_check in search_paths:
        if os.path.exists(path_to_check):
            filepath_found = os.path.abspath(path_to_check)
            logger.info(f"Found model settings file at: {filepath_found}")
            break

    if not filepath_found:
        logger.warning(f"'{MODEL_SETTINGS_FILENAME}' not found in search paths: {search_paths}. Using default LLM client behaviors.")
        return {}

    try:
        with open(filepath_found, 'r') as f:
            model_config = yaml.safe_load(f)
        if model_config is None: # File is empty
            logger.warning(f"'{filepath_found}' is empty. No model settings loaded.")
            return {}
        logger.info(f"Successfully loaded model settings from '{filepath_found}'.")
        return model_config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML from '{filepath_found}': {e}")
        return {}
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading '{filepath_found}': {e}")
        return {}

if __name__ == '__main__':
    # For testing the loader
    print("Attempting to load model settings...")
    # Assuming model_settings.yaml is in the project root when running this test
    # Adjust path if necessary for your test environment
    settings = load_model_settings()
    if settings:
        print("Model settings loaded successfully:")
        import json
        print(json.dumps(settings, indent=2))
    else:
        print("No model settings loaded or file not found.")

    # Test with a specific (likely non-existent) directory
    # print("\nAttempting to load from a non-existent directory:")
    # non_existent_settings = load_model_settings(config_dir="non_existent_path")
    # if not non_existent_settings:
    #     print("Correctly returned empty for non-existent path.")

    # To test from a script in a subdirectory, you might run this from, e.g., project_root/cli:
    # cd .. (to project root)
    # python -m tradingagents.config.model_config_loader
    # (Make sure tradingagents is in PYTHONPATH or run as a module)
