import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class DataProcessor:
    def __init__(self, config):
        self.config = config
        logger.info("Initialized DataProcessor")
    
    def process_raw_data(self) -> pd.DataFrame:
        """Load and process raw data with comprehensive logging"""
        trn_paths = dict(
            raw_data_path = self.config['data']['raw']['trn_path'],
            X_raman_path = self.config['data']['processed']['trn']['X']['raman_path'],
            X_absorption_path = self.config['data']['processed']['trn']['X']['absorption_path'],
            y_ions_path = self.config['data']['processed']['trn']['y_ions_path']
        )
        vld_paths = dict(
            raw_data_path = self.config['data']['raw']['vld_path'],
            X_raman_path = self.config['data']['processed']['vld']['X']['raman_path'],
            X_absorption_path = self.config['data']['processed']['vld']['X']['absorption_path'],
            y_ions_path = self.config['data']['processed']['vld']['y_ions_path']
        )
        
        for paths in [trn_paths, vld_paths]:
            self.load_data(**paths)
    
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