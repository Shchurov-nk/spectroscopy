import yaml
from pathlib import Path
# from typing import Dict, Any, List
import logging

class Config:
    """Configuration management class with validation"""
    
    def __init__(self):
        self.config_path = Path("configs/config.yaml")
        self.data = {}
        self._load_config()
        self._validate_config()
        self.y_ions_path = self.data['data']['processed']['trn']['y_ions_path']
        self.raman_paths = (
            self.data['data']['processed']['trn']['X']['raman_path'], 
            self.data['data']['interim']['raman']['XX_path'],
            self.data['data']['interim']['raman']['Xy_path'],
            self.data['data']['interim']['raman']['masks']['fcbf_path']
            )
        self.absorp_paths = (
            self.data['data']['processed']['trn']['X']['absorption_path'],
            self.data['data']['interim']['absorption']['XX_path'],
            self.data['data']['interim']['absorption']['Xy_path'],
            self.data['data']['interim']['absorption']['masks']['fcbf_path']
            )
        self.level_XX = self.data['feature_selection']['fcbf']['level_XX']
        self.level_Xy = self.data['feature_selection']['fcbf']['level_Xy']
        
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