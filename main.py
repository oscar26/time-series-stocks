from LoadDataAPI import LoadDataAPI
from naivepredictor import NaivePredictor


def main():

    loadDataAPI=LoadDataAPI()
    dataFrameList=loadDataAPI.beginLoadData()

    stocks = ['AAPL','AMZN','BABA','BAC','FB','NKE','NVDA','TSLA','IBM','MSFT']

    for stock in dataFrameList:
        if stock==stocks[0]:
            columns_to_standardize = ['Volume', 'Open', 'High', 'Low', 'Close']
            columns_to_windowize = [2, 3, 4, 5, 6]
            input_window_size = 5
            dataset_path = "datasets/AAPL.csv"
            data=dataFrameList[stock]
            predictor = NaivePredictor(dataset_path,
                columns_to_standardize=columns_to_standardize,
                columns_to_windowize=columns_to_windowize,
                input_window_size=input_window_size,
                data=data)
            predictor.compile_model()
            predictor.test_model(n_splits=9, epochs=90, verbose=0)
            print(predictor.train_results)
            print(predictor.test_results)


if __name__ == '__main__':
	main()
