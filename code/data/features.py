import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class FeatureSelector:
    def __init__(self, config):
        self.config = config
        logger.info("Initialized FeatureSelector")
    
    def get_XX_Xy_correlations(self, df_X, df_y, XX_path, Xy_path):
        corr_XX = df_X.corr().abs()
        corr_Xy = df_X.corrwith(df_y).abs()
        corr_XX.to_feather(XX_path)
        corr_Xy.to_feather(Xy_path)
        logger.info("Successfully computed and saved correlations")

    def compute_correlations(self):
        try:
            logger.info("Loading processed training data for feature selection")
            y_ions_path = self.config['processed']['trn']['y_ions_path']
            df_y = pd.read_feather(y_ions_path)
            for df_X_path, XX_Xy_paths in [
                (self.config['processed']['trn']['X_raman_path'], self.config['interim']['raman']),
                (self.config['processed']['trn']['X_absorption_path'], self.config['interim']['absorption'])
            ]:
                df_X = pd.read_feather(df_X_path)
                XX_path = XX_Xy_paths['XX_path']
                Xy_path = XX_Xy_paths['Xy_path']
                self.get_XX_Xy_correlations(df_X, df_y, XX_path, Xy_path)
        
        except FileNotFoundError as e:
            logger.error(f"Processed data file not found: {e.filename}")
            raise
        except Exception as e:
            logger.error(f"Error processing interim data: {e}", exc_info=True)
            raise