from code.utils.config import load_config
config = load_config()
from code.data.preprocess import DataProcessor

data_processor = DataProcessor(config.data)
data_processor.process_raw_data()