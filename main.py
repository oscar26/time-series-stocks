from LoadDataAPI import LoadDataAPI
from naivepredictor import NaivePredictor

def predictor_1(data):
    columns_to_standardize = ['Volume', 'Open', 'High', 'Low', 'Close']
    columns_to_windowize = [2, 3, 4, 5, 6]
    input_window_size = 5

    predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=data)

    predictor.compile_model()
    predictor.test_model(n_splits=9, epochs=90, verbose=0)
    printData('predictor_1',predictor.train_results,predictor.test_results)

def predictor_2(data):
    df = data.drop('Volume', axis=1)
    columns_to_standardize = ['Open', 'High', 'Low', 'Close']
    columns_to_windowize = [2, 3, 4, 5]
    input_window_size = 5

    predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=df)

    predictor.compile_model()
    predictor.test_model(n_splits=9, epochs=90, verbose=0)
    printData('predictor_2',predictor.train_results,predictor.test_results)

def predictor_3(data):
    df = data.drop('Open', axis=1)
    columns_to_standardize = ['Volume', 'High', 'Low', 'Close']
    columns_to_windowize = [2, 3, 4, 5]
    input_window_size = 5

    predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=df)

    predictor.compile_model()
    predictor.test_model(n_splits=9, epochs=90, verbose=0)
    printData('predictor_3',predictor.train_results,predictor.test_results)

def predictor_4(data):
    df = data.drop('Month', axis=1).drop('Day', axis=1)
    columns_to_standardize = ['Volume', 'Open', 'High', 'Low', 'Close']
    columns_to_windowize = [0, 1, 2, 3, 4]
    input_window_size = 5

    predictor = NaivePredictor(columns_to_standardize=columns_to_standardize,
        columns_to_windowize=columns_to_windowize,
        input_window_size=input_window_size,
        data=df)

    predictor.compile_model()
    predictor.test_model(n_splits=9, epochs=90, verbose=0)
    printData('predictor_4',predictor.train_results,predictor.test_results)

def printData(msn,train_score,test_score):
    print(msn)
    print('Train Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (train_score[0], train_score[1], train_score[2], train_score[3]))
    print('Test Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (test_score[0], test_score[1], test_score[2], test_score[3]))

def main():

    loadDataAPI=LoadDataAPI()
    dataFrameList=loadDataAPI.beginLoadData()
    stocks = ['AAPL','AMZN','BABA','BAC','FB','NKE','NVDA','TSLA','IBM','MSFT']

    for stock in dataFrameList:
        print (" ************************* "+stock+" ******************************\n")
        predictor_1(dataFrameList[stock])
        # predictor_2(dataFrameList[stock])
        # predictor_3(dataFrameList[stock])
        # predictor_4(dataFrameList[stock])



if __name__ == '__main__':
	main()
