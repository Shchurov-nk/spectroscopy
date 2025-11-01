import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        logger.info("Initialized DataProcessor")
    
    def process_raw_data(self) -> pd.DataFrame:
        """Load and process raw data with comprehensive logging"""
        raw_paths = [
            self.config['data']['raw']['trn_path'],
            self.config['data']['raw']['vld_path']
            ]
        processed_paths = [
            self.config['data']['processed']['trn'],
            self.config['data']['processed']['vld']
            ]
        for raw_data_path, processed_data_paths in zip(raw_paths, processed_paths):
            self.load_data(raw_data_path, **processed_data_paths)
    
    def load_data(
            self, 
            raw_data_path: str, 
            X_raman_path: str,
            X_absorption_path: str,
            y_ions_path: str,
            ):
        raman_col_len = self.config['data']['splits']['raman_col_len']
        absorption_col_len = self.config['data']['splits']['absorption_col_len']
        try:
            logger.info(f"Loading raw data from {raw_data_path}")
            df = pd.read_csv(raw_data_path, index_col=0)
            logger.info(f"Successfully loaded raw data: {len(df)} rows, {len(df.columns)} columns")
            
            df_raman = df.iloc[:, :raman_col_len]
            df_raman.to_feather(X_raman_path)
            logger.info(f"Successfully saved Raman: {len(df_raman)} rows, {len(df_raman.columns)} columns")

            df_absorption = df.iloc[:, raman_col_len:raman_col_len+absorption_col_len]
            df_absorption.to_feather(X_absorption_path)
            logger.info(f"Successfully saved Absorption: {len(df_absorption)} rows, {len(df_absorption.columns)} columns")
            
            df_ions = df.iloc[:, raman_col_len+absorption_col_len:]
            df_ions.to_feather(y_ions_path)
            logger.info(f"Successfully saved Ions: {len(df_ions)} rows, {len(df_ions.columns)} columns")

        except FileNotFoundError:
            logger.error(f"Raw data file not found: {raw_data_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing raw data: {e}", exc_info=True)
            raise