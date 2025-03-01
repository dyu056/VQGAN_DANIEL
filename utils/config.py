import json
import os
from typing import Dict, Any

class Config:
    def __init__(self, config_path: str = None):
        # Load default config
        default_config_path = os.path.join(os.path.dirname(__file__), '../configs/default_config.json')
        with open(default_config_path, 'r') as f:
            self.config = json.load(f)
            
        # Override with custom config if provided
        if config_path is not None:
            with open(config_path, 'r') as f:
                custom_config = json.load(f)
                self._update_config(self.config, custom_config)
    
    def _update_config(self, default: Dict[str, Any], custom: Dict[str, Any]) -> None:
        """Recursively update default config with custom values."""
        for key, value in custom.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._update_config(default[key], value)
            else:
                default[key] = value
    
    def __getattr__(self, name: str) -> Any:
        """Allow accessing config values as attributes."""
        if name in self.config:
            return self.config[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")
    
    def save(self, path: str) -> None:
        """Save current config to file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=4) 