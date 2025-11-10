import logging
import pandas as pd

from code.utils.config import load_config
from code.data.dataset import DataProcessor
from code.data.features import FeatureSelector

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

config = load_config()

data_processor = DataProcessor(config)
data_processor.process_raw_data()

raman_paths = (
    config['data']['processed']['trn']['X']['raman_path'], 
    config['data']['interim']['raman']['XX_path'],
    config['data']['interim']['raman']['Xy_path']
    )
absorp_paths = (
    config['data']['processed']['trn']['X']['absorption_path'],
    config['data']['interim']['absorption']['XX_path'],
    config['data']['interim']['absorption']['Xy_path']
    )
level_XX = config['feature_selection']['fcbf']['level_XX']
level_Xy = config['feature_selection']['fcbf']['level_Xy']

try:
    logger.info("Loading processed training data for feature selection")
    y_ions_path = config['data']['processed']['trn']['y_ions_path']
    df_y = pd.read_feather(y_ions_path)
    for df_X_path, XX_path, Xy_path in (raman_paths, absorp_paths):
        df_X = pd.read_feather(df_X_path)
        feature_selector = FeatureSelector(df_X, df_y)
        corr_XX, corr_Xy = feature_selector.calculate_correlations()
        corr_XX.to_feather(XX_path)
        corr_Xy.to_feather(Xy_path)

        mask = feature_selector.fcbf(level_XX, level_Xy)

except FileNotFoundError as e:
    logger.error(f"Processed data file not found: {e.filename}")
    raise
except Exception as e:
    logger.error(f"Error processing data: {e}", exc_info=True)
    raise

