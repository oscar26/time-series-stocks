from LoadDataAPI import LoadDataAPI
from naivepredictor import NaivePredictor
from transformpredictor import TransformPredictor
import json

def predictor_1(data, is_production=False, stock_name=None):
    columns_to_standardize = ['Volume', 'Open', 'High', 'Low', 'Close']
    columns_to_windowize = [2, 3, 4, 5, 6]
    input_window_size = 5
    predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=data)
    predictor.compile_model()
    if is_production:
        predictor.fit_model(epochs=90)
        save_model(predictor, 'predictor_1_' + stock_name)
    else:
        predictor.test_model(n_splits=9, epochs=90, verbose=0)
        print_data('predictor_1',predictor.train_results,predictor.test_results)

def predictor_2(data, is_production=False, stock_name=None):
    df = data.drop('Volume', axis=1)
    columns_to_standardize = ['Open', 'High', 'Low', 'Close']
    columns_to_windowize = [2, 3, 4, 5]
    input_window_size = 5
    predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=df)
    predictor.compile_model()
    if is_production:
        predictor.fit_model(epochs=90)
        save_model(predictor, 'predictor_2_' + stock_name)
    else:
        predictor.test_model(n_splits=9, epochs=90, verbose=0)
        print_data('predictor_2',predictor.train_results,predictor.test_results)

def predictor_3(data, is_production=False, stock_name=None):
    df = data.drop('Open', axis=1)
    columns_to_standardize = ['Volume', 'High', 'Low', 'Close']
    columns_to_windowize = [2, 3, 4, 5]
    input_window_size = 5
    predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=df)
    predictor.compile_model()
    if is_production:
        predictor.fit_model(epochs=90)
        save_model(predictor, 'predictor_3_' + stock_name)
    else:
        predictor.test_model(n_splits=9, epochs=90, verbose=0)
        print_data('predictor_3',predictor.train_results,predictor.test_results)

def predictor_4(data, is_production=False, stock_name=None):
    df = data.drop('Month', axis=1).drop('Day', axis=1)
    columns_to_standardize = ['Volume', 'Open', 'High', 'Low', 'Close']
    columns_to_windowize = [0, 1, 2, 3, 4]
    input_window_size = 5
    predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=df)
    predictor.compile_model()
    if is_production:
        predictor.fit_model(epochs=90)
        save_model(predictor, 'predictor_4_' + stock_name)
    else:
        predictor.test_model(n_splits=9, epochs=90, verbose=0)
        print_data('predictor_4',predictor.train_results,predictor.test_results)

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
        save_model(predictor, 'predictor_5_' + stock_name)
    else:
        predictor.test_model(n_splits=9, epochs=100, verbose=0)
        print_data('predictor_5', predictor.train_results, predictor.test_results)

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
        save_model(predictor, 'predictor_6_' + stock_name)
    else:
        predictor.test_model(n_splits=9, epochs=100, verbose=0)
        print_data('predictor_6', predictor.train_results, predictor.test_results)

def print_data(msn, train_score, test_score):
    print(msn)
    print('Train Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (train_score[0], train_score[1], train_score[2], train_score[3]))
    print('Test Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (test_score[0], test_score[1], test_score[2], test_score[3]))

def save_model(predictor, file_name):
    # Guardado de la red neuronal
    predictor.model.save('saved_models/' + file_name + '.h5')
    # Guardado de metadatos sobre el predictor
    dataset_path = 'saved_models/' + file_name + '_input_data.pkl'
    predictor.data.to_pickle(dataset_path)
    metadata = {
        'input_window_size': predictor.input_window_size,
        'columns_to_windowize': predictor.columns_to_windowize,
        'columns_to_standardize': predictor.columns_to_standardize,
        'column_means': predictor.column_means,
        'column_stds': predictor.column_stds,
        'dataset_path': dataset_path
    }
    metadata_path = 'saved_models/' + file_name + '_metadata.json'
    with open(metadata_path, 'w') as fp:
        json.dump(metadata, fp)

def run_tests(stocks_data):
    for stock in stocks_data:
        print (" ************************* " + stock + " ******************************\n")
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

def main():
    # Stocks: ['AAPL','AMZN','BABA','BAC','FB','NKE','NVDA','TSLA','IBM','MSFT']
    isTesting = False
    loadDataAPI = LoadDataAPI()
    stocks_data = loadDataAPI.beginLoadData()
    if isTesting:
        run_tests(stocks_data)
    else:
        train_production_models(stocks_data)

if __name__ == '__main__':
	main()