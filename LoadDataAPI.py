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
        print(data_file[:50])
        #print(data_file.tail())

    # Guardamos en un archivo csv, los datos de un DataFrame en un path especifico.
    def save_data(self,path,df):
        np.savetxt(path, df.as_matrix(), delimiter=',')


    def preprocess_data(self,data_file):
        """Preprocesamiento del conjunto de datos."""
        df = data_file.reset_index()
        date = pd.to_datetime(df['Date'])
        df.insert(0, 'Month', date.dt.month)
        df.insert(1, 'Day', date.dt.day)
        df = df.drop('Date', axis=1)
        new_column_order = ['Month', 'Day', 'Volume', 'Open', 'High', 'Low', 'Close']
        data = df.reindex(columns=new_column_order)
        return data

    def beginLoadData(self):
        loadDataApi=LoadDataAPI();
        # Abreviatura de algunas Empresas
        stocks = ['AAPL','AMZN','BABA','BAC','FB','NKE','NVDA','TSLA','IBM','MSFT']
        # Google Finance API
        API='google'
        dataFrameList={}

        # Intervalo de tiempo para carga de los datos[start-end]
        #Fecha de Inicio
        start = datetime.datetime(2009, 1, 1)
        #Fecha Fin
        end = datetime.datetime(2017, 6, 3)

        for stock in stocks:
            save_file_path = 'datasets/'+stock+'business.csv'
            data_file=loadDataApi.loadDataFromAPI(stock,start,end,API)
            data_proccess=loadDataApi.preprocess_data(data_file)
            # loadDataApi.printData(data_file)

            loadDataApi.save_data(save_file_path,data_proccess)
            dataFrameList[stock]=data_proccess

        print("\t ***   Finalized  data load    ***\n")

        return dataFrameList

def main():
    loadDataApi=LoadDataAPI()
    loadDataApi.beginLoadData()

if __name__ == '__main__':
	main()
