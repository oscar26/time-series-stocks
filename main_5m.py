import numpy as np
import pandas as pd
from LoadDataAPI import LoadDataAPI
from naivepredictor import NaivePredictor
from transformpredictor import TransformPredictor

dataset_type = '5m'

def predictor_1(data):
	extended_data = data.copy()
	columns_to_standardize = ['Tick Range', 'Open', 'High', 'Low', 'Close']
	columns_to_windowize = [2, 3, 4, 5, 6]
	input_window_size = 10
	predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=extended_data)
	predictor.compile_model()
	predictor.test_model(n_splits=9, epochs=90, verbose=0)
	print_data(''.join(('predictor_1_', dataset_type)), predictor.train_results, predictor.test_results)

def predictor_2(data):
	pass

def predictor_3(data):
	pass

def predictor_4(data):
	pass

def predictor_5(data):
	pass

def predictor_6(data):
	pass

def print_data(msn, train_score, test_score):
    print(msn)
    print('Train Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (train_score[0], train_score[1], train_score[2], train_score[3]))
    print('Test Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (test_score[0], test_score[1], test_score[2], test_score[3]))

def run_tests(stocks_data):
    for stock in stocks_data:
        print("*** " + stock + " ***")
        predictor_1(stocks_data[stock])
        # predictor_2(stocks_data[stock])
        # predictor_3(stocks_data[stock])
        # predictor_4(stocks_data[stock])
        # predictor_5(stocks_data[stock])
        # predictor_6(stocks_data[stock])

def extract_hour_minute(stocks_data):
	new_stocks_data = {}
	for stock in stocks_data:
		data = stocks_data[stock].copy()
		data['Hour'] = (data['Date'].dt.hour - 8).astype(np.int32)
		data['Minute'] = data['Date'].dt.minute.astype(np.int32)
		data = data[['Hour', 'Minute', 'Tick Range', 'Open', 'High', 'Low', 'Close']]
		new_stocks_data[stock] = data.dropna().reset_index(drop=True)
	return new_stocks_data

def main():
    # Stocks: ['AAPL','AMZN','BABA','BAC','FB','NKE','NVDA','TSLA']
    loadDataAPI = LoadDataAPI()
    stocks_data = loadDataAPI.load_clean_5m_data()
    stocks_data = extract_hour_minute(stocks_data)
    run_tests(stocks_data)

if __name__ == '__main__':
	main()