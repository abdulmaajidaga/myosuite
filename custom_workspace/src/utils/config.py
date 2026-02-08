"""
Centralized configuration loader for the stroke rehabilitation pipeline.
All scripts should use this module instead of hardcoding paths.
"""
import os
import yaml

_config = None
_project_root = None


def get_project_root():
    """Returns the absolute path to the project root (custom_workspace/)."""
    global _project_root
    if _project_root is None:
        # Walk up from this file: src/utils/config.py -> src/utils -> src -> project_root
        _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return _project_root


def load_config():
    """Load and return the full config dictionary from settings.yaml."""
    global _config
    if _config is None:
        config_path = os.path.join(get_project_root(), "config", "settings.yaml")
        with open(config_path, "r") as f:
            _config = yaml.safe_load(f)
    return _config


def get_path(key):
    """
    Get an absolute path from the config's paths section.

    Usage:
        model_path = get_path("mujoco_arm_model")
    """
    cfg = load_config()
    relative = cfg["paths"][key]
    return os.path.join(get_project_root(), relative)


def get(section, key=None):
    """
    Get a config value from any section.

    Usage:
        data_rate = get("pipeline", "data_rate")
        cvae_cfg = get("cvae")  # returns entire section dict
    """
    cfg = load_config()
    if key is None:
        return cfg[section]
    return cfg[section][key]
