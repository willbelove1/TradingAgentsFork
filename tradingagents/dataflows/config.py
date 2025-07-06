import tradingagents.default_config as default_config
from tradingagents.config.model_config_loader import load_model_settings # Import new loader
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Use default config but allow it to be overridden
_config: Optional[Dict] = None
DATA_DIR: Optional[str] = None

def _deep_merge_dicts(base: Dict, new: Dict) -> Dict:
    """
    Recursively merges 'new' dict into 'base' dict.
    If a key exists in both and both values are dicts, it recursively merges them.
    Otherwise, the value from 'new' overwrites the value in 'base'.
    """
    merged = base.copy()
    for key, new_val in new.items():
        if key in merged:
            base_val = merged[key]
            if isinstance(base_val, dict) and isinstance(new_val, dict):
                merged[key] = _deep_merge_dicts(base_val, new_val)
            else: # new_val overrides base_val
                merged[key] = new_val
        else:
            merged[key] = new_val
    return merged

def initialize_config(custom_model_settings_path: Optional[str] = None):
    """
    Initialize the configuration with default values, then load and merge model settings.
    Args:
        custom_model_settings_path (str, optional): Path to a custom model_settings.yaml.
                                                    If None, uses default search paths.
    """
    global _config, DATA_DIR
    if _config is None:
        base_config = default_config.DEFAULT_CONFIG.copy()

        # Load model settings
        # Pass custom_model_settings_path if provided, else loader uses its defaults
        model_specific_settings = load_model_settings(config_dir=os.path.dirname(custom_model_settings_path) if custom_model_settings_path else None)

        if model_specific_settings:
            logger.info("Merging model-specific settings into default configuration.")
            # Merge model settings into base config. Model settings can override defaults.
            # Using a simple update for top-level keys from model_settings.yaml like llm_provider, default_text_model etc.
            # For nested structures like 'agent_model_configs', simple update also works well.
            # If deeper merging of sub-dictionaries is needed for other keys, a deep_merge function would be required.
            # For now, model_settings.yaml is expected to define its structure, and its keys will override default_config.
            _config = _deep_merge_dicts(base_config, model_specific_settings)
        else:
            logger.info("No model-specific settings loaded or file was empty. Using default configuration values for models.")
            _config = base_config

        DATA_DIR = _config.get("data_dir") # Use .get for safety
        if DATA_DIR is None:
            logger.warning("'data_dir' not found in the final configuration.")

def set_config(new_config_values: Dict):
    """
    Update the global configuration with new values.
    This will re-initialize if _config is None, then update.
    """
    global _config, DATA_DIR
    if _config is None:
        initialize_config() # Ensure base and model configs are loaded

    # We should merge the new_config_values into the existing _config
    # This allows partial updates without losing previously loaded model settings or defaults.
    _config = _deep_merge_dicts(_config, new_config_values)

    new_data_dir = _config.get("data_dir")
    if new_data_dir:
        DATA_DIR = new_data_dir
    logger.info("Global configuration updated with new values.")


def get_config() -> Dict:
    """Get the current global configuration."""
    if _config is None:
        initialize_config()
    return _config.copy() # Return a copy to prevent external modification


# Initialize with default config and attempt to load model_settings.yaml from default locations
import os # Add os import for dirname if not already present
initialize_config()
