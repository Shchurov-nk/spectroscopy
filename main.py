import logging
import pandas as pd

from spectroscopy.config import Config
from spectroscopy.dataset import DataProcessor
from spectroscopy.features import FeatureSelector

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

config = Config()

data_processor = DataProcessor(config.data)
data_processor.process_raw_data()

try:
    logger.info("Loading processed training data for feature selection")
    df_y = pd.read_feather(config.y_ions_path)
    data_paths = (config.raman_paths, config.absorp_paths)
    for df_X_path, XX_path, Xy_path, fcbf_mask_path in data_paths:
        df_X = pd.read_feather(df_X_path)
        feature_selector = FeatureSelector(df_X, df_y)
        feature_selector.calculate_correlations()
        feature_selector.save_correlations(XX_path, Xy_path)
        feature_selector.fcbf(config.level_XX, config.level_Xy)
        feature_selector.save_mask(fcbf_mask_path)

except FileNotFoundError as e:
    logger.error(f"Processed data file not found: {e.filename}")
    raise
except Exception as e:
    logger.error(f"Error processing data: {e}", exc_info=True)
    raise

