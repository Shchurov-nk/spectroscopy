from code.utils.config import load_config
config = load_config()
from code.data.dataset import DataProcessor
from code.data.features import FeatureSelector

data_processor = DataProcessor(config.data)
data_processor.process_raw_data()

feature_selector = FeatureSelector(config.data)
feature_selector.calculate_correlations()