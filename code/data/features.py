import logging
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

class FeatureSelector:
    def __init__(self, config):
        self.config = config
        logger.info("Initialized FeatureSelector")

    def calculate_correlations(self):
        """
        Calculate absolute correlation values between:
        - input features (XX)
        - input features and targets (Xy)
        """
        try:
            logger.info("Loading processed training data for feature selection")
            y_ions_path = self.config['data']['processed']['trn']['y_ions_path']
            df_y = pd.read_feather(y_ions_path)

            for df_X_path, XX_path, Xy_path in [
                (
                    self.config['data']['processed']['trn']['X']['raman_path'], 
                    self.config['data']['interim']['raman']['XX_path'],
                    self.config['data']['interim']['raman']['Xy_path']
                ),
                (
                    self.config['data']['processed']['trn']['X']['absorption_path'], 
                    self.config['data']['interim']['absorption']['XX_path'],
                    self.config['data']['interim']['absorption']['Xy_path']
                )
            ]:
                df_X = pd.read_feather(df_X_path)

                corr_XX = df_X.corr().abs()
                corr_Xy = df_X.corrwith(df_y).abs().to_frame()
                
                corr_XX.to_feather(XX_path)
                corr_Xy.to_feather(Xy_path)
                logger.info("Successfully calculated and saved correlations")
        
        except FileNotFoundError as e:
            logger.error(f"Processed data file not found: {e.filename}")
            raise
        except Exception as e:
            logger.error(f"Error processing interim data: {e}", exc_info=True)
            raise