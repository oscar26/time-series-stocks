from LoadDataAPI import LoadDataAPI
from naivepredictor import NaivePredictor
from transformpredictor import TransformPredictor
from modelloader import ModelLoader

def predictor_1(data, is_production=False, stock_name=None):
    columns_to_standardize = ['Volume', 'Open', 'High', 'Low', 'Close']
    columns_to_windowize = [2, 3, 4, 5, 6]
    input_window_size = 5
    if is_production:
        predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
            columns_to_windowize=columns_to_windowize,
            input_window_size=input_window_size,
            data=data,
            is_production=is_production)
        predictor.compile_model()
        predictor.fit_model(epochs=90)
        predictor.save_predictor('predictor_1_' + dataset_type, stock_name)
    else:
        predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
            columns_to_windowize=columns_to_windowize,
            input_window_size=input_window_size,
            data=data)
        predictor.compile_model()
        predictor.test_model(n_splits=9, epochs=90, verbose=0)
        print_data('predictor_1_' + dataset_type, predictor.train_results, predictor.test_results)

def predictor_2(data, is_production=False, stock_name=None):
    df = data.drop('Volume', axis=1)
    columns_to_standardize = ['Open', 'High', 'Low', 'Close']
    columns_to_windowize = [2, 3, 4, 5]
    input_window_size = 5
    if is_production:
        predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
            columns_to_windowize=columns_to_windowize,
            input_window_size=input_window_size,
            data=df,
            is_production=is_production)
        predictor.compile_model()
        predictor.fit_model(epochs=90)
        predictor.save_predictor('predictor_2_' + dataset_type, stock_name)
    else:
        predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
            columns_to_windowize=columns_to_windowize,
            input_window_size=input_window_size,
            data=df)
        predictor.compile_model()
        predictor.test_model(n_splits=9, epochs=90, verbose=0)
        print_data('predictor_2_' + dataset_type, predictor.train_results, predictor.test_results)

def predictor_3(data, is_production=False, stock_name=None):
    df = data.drop('Open', axis=1)
    columns_to_standardize = ['Volume', 'High', 'Low', 'Close']
    columns_to_windowize = [2, 3, 4, 5]
    input_window_size = 5
    if is_production:
        predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
            columns_to_windowize=columns_to_windowize,
            input_window_size=input_window_size,
            data=df,
            is_production=is_production)
        predictor.compile_model()
        predictor.fit_model(epochs=90)
        predictor.save_predictor('predictor_3_' + dataset_type, stock_name)
    else:
        predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
            columns_to_windowize=columns_to_windowize,
            input_window_size=input_window_size,
            data=df)
        predictor.compile_model()
        predictor.test_model(n_splits=9, epochs=90, verbose=0)
        print_data('predictor_3_' + dataset_type, predictor.train_results, predictor.test_results)

def predictor_4(data, is_production=False, stock_name=None):
    df = data.drop('Month', axis=1).drop('Day', axis=1)
    columns_to_standardize = ['Volume', 'Open', 'High', 'Low', 'Close']
    columns_to_windowize = [0, 1, 2, 3, 4]
    input_window_size = 5
    if is_production:
        predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
            columns_to_windowize=columns_to_windowize,
            input_window_size=input_window_size,
            data=df,
            is_production=is_production)
        predictor.compile_model()
        predictor.fit_model(epochs=90)
        predictor.save_predictor('predictor_4_' + dataset_type, stock_name)
    else:
        predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
            columns_to_windowize=columns_to_windowize,
            input_window_size=input_window_size,
            data=df)
        predictor.compile_model()
        predictor.test_model(n_splits=9, epochs=90, verbose=0)
        print_data('predictor_4_' + dataset_type, predictor.train_results, predictor.test_results)

def predictor_5(data, is_production=False, stock_name=None):
    df = data.drop('Volume', axis=1).drop('Open', axis=1).drop('High', axis=1).drop('Low', axis=1)
    columns_to_standardize = []
    columns_to_windowize = [2]
    input_window_size = 20
    predictor = TransformPredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=df)
    predictor.compile_model()
    if is_production:
        predictor.fit_model(epochs=100)
        predictor.save_predictor('predictor_5_' + dataset_type, stock_name)
    else:
        predictor.test_model(n_splits=9, epochs=100, verbose=0)
        print_data('predictor_5_' + dataset_type, predictor.train_results, predictor.test_results)

def predictor_6(data, is_production=False, stock_name=None):
    df = data.drop('Volume', axis=1).drop('Open', axis=1).drop('High', axis=1).drop('Low', axis=1).drop('Month', axis=1).drop('Day', axis=1)
    columns_to_standardize = []
    columns_to_windowize = [0]
    input_window_size = 20
    predictor = TransformPredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=df)
    predictor.compile_model()
    if is_production:
        predictor.fit_model(epochs=100)
        predictor.save_predictor('predictor_6_' + dataset_type, stock_name)
    else:
        predictor.test_model(n_splits=9, epochs=100, verbose=0)
        print_data('predictor_6_' + dataset_type, predictor.train_results, predictor.test_results)

def print_data(msn, train_score, test_score):
    print(msn)
    print('Train Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (train_score[0], train_score[1], train_score[2], train_score[3]))
    print('Test Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (test_score[0], test_score[1], test_score[2], test_score[3]))

def run_tests(stocks_data):
    for stock in stocks_data:
        print(" ************************* " + stock + " ******************************")
        predictor_1(stocks_data[stock])
        predictor_2(stocks_data[stock])
        predictor_3(stocks_data[stock])
        predictor_4(stocks_data[stock])
        predictor_5(stocks_data[stock])
        predictor_6(stocks_data[stock])

def train_production_models(stocks_data):
    for stock in stocks_data:
        print ("\nSaving models for " + stock + "\n")
        predictor_1(stocks_data[stock], True, stock)
        predictor_2(stocks_data[stock], True, stock)
        predictor_3(stocks_data[stock], True, stock)
        predictor_4(stocks_data[stock], True, stock)
        predictor_5(stocks_data[stock], True, stock)
        predictor_6(stocks_data[stock], True, stock)

dataset_types = ['5m', '1d']
dataset_type = '1d'

def main():
    # Stocks: ['AAPL','AMZN','BABA','BAC','FB','NKE','NVDA','TSLA','IBM','MSFT']
    isTesting = False
    loadDataAPI = LoadDataAPI()
    if dataset_type == '1d':
        stocks_data = loadDataAPI.beginLoadData()
    if dataset_type == '5m':
        # Decided to create a main method exclusively for the 5 minutes data.
        pass
    if isTesting:
        run_tests(stocks_data)
    else:
        train_production_models(stocks_data)
        loader = ModelLoader()
        loader.predict_batch_next_day()

if __name__ == '__main__':
	main()