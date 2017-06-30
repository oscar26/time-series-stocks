import numpy as np
import pandas as pd
import json
from datetime import datetime, timedelta
from itertools import chain
from keras.models import load_model
from naivepredictor import NaivePredictor
from transformpredictor import TransformPredictor

class ModelLoader(object):
	def load_predictor(self, name, symbol, predictor_type, name_delimiter='_'):
		# Load metadata
		metadata_path = 'saved_models/' + name + name_delimiter + symbol + '_metadata.json'
		model_path = 'saved_models/' + name + name_delimiter + symbol + '.h5'
		with open(metadata_path, 'r') as fp:
			metadata = json.load(fp)
		data = pd.read_pickle(metadata['dataset_path'])
		if predictor_type == 'NaivePredictor':
			predictor = NaivePredictor(columns_to_standardize=metadata['columns_to_standardize'], columns_to_windowize=metadata['columns_to_windowize'], input_window_size=metadata['input_window_size'], data=data)
		if predictor_type == 'TransformPredictor':
			predictor = TransformPredictor(columns_to_standardize=metadata['columns_to_standardize'], columns_to_windowize=metadata['columns_to_windowize'], input_window_size=metadata['input_window_size'], data=data)
			predictor.last_price = metadata['last_price']
		predictor.column_means = metadata['column_means']
		predictor.column_stds = metadata['column_stds']
		predictor.model = load_model(model_path)
		predictor.name = metadata['name']
		predictor.symbol = metadata['symbol']
		return predictor

	def predict_batch_next_day(self):
		dataset_type = '1d'
		stocks = ['AAPL','AMZN','BABA','BAC','FB','NKE','NVDA','TSLA','IBM','MSFT']
		predictor_names = ['predictor_1', 'predictor_2', 'predictor_3', 'predictor_4', 'predictor_5', 'predictor_6']
		columns = list(chain.from_iterable((''.join((q, '_lower')), ''.join((q, '_close')), ''.join((q, '_upper'))) for q in stocks))
		table = pd.DataFrame(np.zeros((len(predictor_names), len(columns))), index=predictor_names, columns=columns)
		for name in predictor_names:
			predictor_name = ''.join((name, '_', dataset_type))
			print(predictor_name)
			for stock in stocks:
				print(''.join(('  ', stock)), end='')
				if '5' in predictor_name or '6' in predictor_name:
					predictor = self.load_predictor(predictor_name, stock, 'TransformPredictor')
				else:
					predictor = self.load_predictor(predictor_name, stock, 'NaivePredictor')
				prediction = predictor.predict()
				table.loc[name, ''.join((stock, '_lower'))] = prediction[1]
				table.loc[name, ''.join((stock, '_close'))] = prediction[0]
				table.loc[name, ''.join((stock, '_upper'))] = prediction[2]
				print(': %.2f / %.2f / %.2f' % (prediction[1], prediction[0], prediction[2]))
		result_summary = table.mean()
		result_summary.name = 'Mean'
		table = table.append(result_summary)
		date = str((datetime.today() + timedelta(days=1)).date())
		table.to_csv(''.join(('results/', date, '_pred.csv')), float_format='%.2f')
		print(table)

def test_model():
	dataset_type = '1d'
	predictor_name = 'predictor_6_' + dataset_type
	stock = 'AAPL'
	loader = ModelLoader()
	if '5' in predictor_name or '6' in predictor_name:
		predictor = loader.load_predictor(predictor_name, stock, 'TransformPredictor')
	else:
		predictor = loader.load_predictor(predictor_name, stock, 'NaivePredictor')
	prediction = predictor.predict()
	print(prediction)

if __name__ == '__main__':
	loader = ModelLoader()
	loader.predict_batch_next_day()
	# test_model()