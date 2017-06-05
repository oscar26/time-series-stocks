import pandas as pd
import json
from keras.models import load_model
from naivepredictor import NaivePredictor
from transformpredictor import TransformPredictor

class ModelLoader(object):
	def load_predictor(self, predictor_name, stock, predictor_type):
		# Load metadata
		metadata_path = 'saved_models/' + predictor_name + stock + '_metadata.json'
		model_path = 'saved_models/' + predictor_name + stock + '.h5'
		with open(metadata_path, 'r') as fp:
			metadata = json.load(fp)
		data = pd.read_pickle(metadata['dataset_path'])
		if predictor_type == 'NaivePredictor':
			predictor = NaivePredictor(columns_to_standardize=metadata['columns_to_standardize'], columns_to_windowize=metadata['columns_to_windowize'], input_window_size=metadata['input_window_size'], data=data)
		if predictor_type == 'TransformPredictor':
			predictor = TransformPredictor(columns_to_standardize=metadata['columns_to_standardize'], columns_to_windowize=metadata['columns_to_windowize'], input_window_size=metadata['input_window_size'], data=data)
		predictor.column_means = metadata['column_means']
		predictor.column_stds = metadata['column_stds']
		predictor.model = load_model(model_path)
		predictor.name = predictor_name
		predictor.stock = stock
		return predictor

def test():
	loader = ModelLoader()
	stocks = ['AAPL','AMZN','BABA','BAC','FB','NKE','NVDA','TSLA','IBM','MSFT']
	predictor_names = ['predictor_1_', 'predictor_2_', 'predictor_3_', 'predictor_4_', 'predictor_5_', 'predictor_6_']
	for predictor_name in predictor_names:
		print(predictor_name)
		for stock in stocks:
			print('  ' + stock)
			if predictor_name == 'predictor_5_' or predictor_name == 'predictor_6_':
				predictor = loader.load_predictor(predictor_name, stock, 'TransformPredictor')
			else:
				predictor = loader.load_predictor(predictor_name, stock, 'NaivePredictor')

if __name__ == '__main__':
	test()