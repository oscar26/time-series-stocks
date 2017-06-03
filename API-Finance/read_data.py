import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from os.path import isfile


class LoadDataAPI(object):


    def loadDataFromAPI(self,stock, start,end,API):
        #Carga de datos de  Google Finance API
        data_file = web.DataReader(stock, API, start, end)
        return data_file

    def printData(self,data_file):
        print(data_file[50:])
        print(data_file.tail())

    # Guardamos en un archivo csv, los datos de un DataFrame en un path especifico.
    def save_data(self,path,df):
        np.savetxt(path, df.as_matrix(), delimiter=',')


    def windowize_series(self, data, size=2, horizon=1, column_indexes=None):
        '''Last column of the dataframe must be the target Y.
    	'''
        if type(data).__module__ != np.__name__:
            raise('The data must be a numpy array.')
        if data is None or data.size == 0:
            raise('The array is none or empty.')

        num_windows = data.shape[0] -size - horizon + 1
        if column_indexes is None:
            input_vector_length = data.shape[1] * size
        else:
            input_vector_length = size*len(column_indexes) + data.shape[1] - len(column_indexes)
            all_indexes = range(data.shape[1])
            ignored_indexes = [index for index in all_indexes if index not in column_indexes]


        X = np.zeros((num_windows, input_vector_length))
        Y = data[size+horizon-1:, data.shape[1]-1]

        for i in range(num_windows):
            if column_indexes is None:
                X[i, :] = np.reshape(data[i:i+size, :], size*data.shape[1], order='F')
            else:
                input_vector = np.reshape(data[i:i+size, column_indexes], input_vector_length - len(ignored_indexes), order='F')
                input_vector = np.insert(input_vector, 0, data[i, ignored_indexes])
                X[i, :] = input_vector
        return X, Y


def main():
    loadDataApi=LoadDataAPI();
    # Abreviatura de algunas Empresas
    stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
    # Google Finance API
    API='google'

    save_file_path = 'datasets/df-business.csv'
    # Intervalo de tiempo para carga de los datos[start-end]
    #Fecha de Inicio
    start = datetime.datetime(2009, 1, 1)
    #Fecha Fin
    end = datetime.datetime(2017, 6, 3)

    data_file=loadDataApi.loadDataFromAPI(stocks[0],start,end,API)
    loadDataApi.printData(data_file)

    loadDataApi.save_data(save_file_path,data_file)

    X,Y=loadDataApi.windowize_series(data_file.as_matrix(),4,1)

    print(X)

if __name__ == '__main__':
	main()
