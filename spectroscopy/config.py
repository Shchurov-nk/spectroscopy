import yaml
from pathlib import Path
# from typing import Dict, Any, List
import logging

class Config:
    """Configuration management class with validation"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config_path = Path(config_path)
        self.data = {}
        self._load_config()
        self._validate_config()
        
    def _load_config(self) -> None:
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r') as file:
                self.data = yaml.safe_load(file)
            logging.info(f"Successfully loaded config from {self.config_path}")
        except FileNotFoundError:
            logging.error(f"Config file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logging.error(f"Error parsing YAML config: {e}")
            raise
    
    def _validate_config(self) -> None:
        """Validate configuration structure and values"""
        required_sections = []#'data', 'features', 'model', 'output']
        
        for section in required_sections:
            if section not in self.data:
                raise ValueError(f"Missing required config section: {section}")

# Convenience function
def load_config(config_path: str = "configs/config.yaml") -> Config:
    """Load configuration from YAML file"""
    return Config(config_path).data