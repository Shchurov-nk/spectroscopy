import logging
import pandas as pd

from spectroscopy.config import Config
from spectroscopy.dataset import DataProcessor
from spectroscopy.features import FeatureSelector
from spectroscopy.modeling.train import ModelTrainer

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)

config = Config()

data_processor = DataProcessor(config.data)
data_processor.process_raw_data()

try:
    logger.info("Loading processed training data for feature selection")
    y_trn = pd.read_feather(config.y_ions_path)
    data_paths = (config.raman_paths, config.absorp_paths)
    for X_trn_path, XX_path, Xy_path, fcbf_mask_path in data_paths:
        X_trn = pd.read_feather(X_trn_path)
        feature_selector = FeatureSelector(X_trn, y_trn)
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

raman_full_training = {
    'X_trn': config.data['data']['processed']['trn']['X']['raman_path'],
    'y_trn': config.data['data']['processed']['trn']['y_ions_path'],
    'X_vld': config.data['data']['processed']['vld']['X']['raman_path'],
    'y_vld': config.data['data']['processed']['vld']['y_ions_path'],
    'mask': None,
    'model_path': config.data['output']['models']['model_path']
}

raman_fcbf_training = {
    'X_trn': config.data['data']['processed']['trn']['X']['raman_path'],
    'y_trn': config.data['data']['processed']['trn']['y_ions_path'],
    'X_vld': config.data['data']['processed']['vld']['X']['raman_path'],
    'y_vld': config.data['data']['processed']['vld']['y_ions_path'],
    'mask': config.data['data']['interim']['raman']['masks']['fcbf_path']
}

absorption_full_training = {
    'X_trn': config.data['data']['processed']['trn']['X']['absorption_path'],
    'y_trn': config.data['data']['processed']['trn']['y_ions_path'],
    'X_vld': config.data['data']['processed']['vld']['X']['absorption_path'],
    'y_vld': config.data['data']['processed']['vld']['y_ions_path'],
    'mask': None,
    'model_path': config.data['output']['models']['absorption_raw_model_path']
}

absorption_fcbf_training = {
    'X_trn': config.data['data']['processed']['trn']['X']['absorption_path'],
    'y_trn': config.data['data']['processed']['trn']['y_ions_path'],
    'X_vld': config.data['data']['processed']['vld']['X']['absorption_path'],
    'y_vld': config.data['data']['processed']['vld']['y_ions_path'],
    'mask': config.data['data']['interim']['absorption']['masks']['fcbf_path']
}

# TBD: concat raman and absorption features
both_full_training = {}
both_fcbf_training = {}


for training_params in [
    raman_full_training,
    raman_fcbf_training,
    absorption_full_training,
    absorption_fcbf_training,
    # both_full_training
    # both_fcbf_training
]:
    X_trn = pd.read_feather(training_params['X_trn'])
    y_trn = pd.read_feather(training_params['y_trn'])
    X_vld = pd.read_feather(training_params['X_vld'])
    y_vld = pd.read_feather(training_params['y_vld'])
    
    model_params = config.data['model_params']
    model_path = config.data['output']['models']['model_path']
    model_trainer = ModelTrainer()