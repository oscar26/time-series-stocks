import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from os import listdir
from os.path import isfile, join

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


    def preprocess_data(self, data_file):
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
        loadDataApi = LoadDataAPI();
        # Abreviatura de algunas Empresas
        stocks = ['AAPL','AMZN','BABA','BAC','FB','NKE','NVDA','TSLA','IBM','MSFT']
        # Google Finance API
        API = 'google'
        data_frames = {}
        # Intervalo de tiempo para carga de los datos[start-end]
        #Fecha de Inicio
        start = datetime.datetime(2009, 1, 1)
        #Fecha Fin
        end = datetime.datetime.today()
        for stock in stocks:
            save_file_path = 'datasets/' + stock + 'business.csv'
            data_file = loadDataApi.loadDataFromAPI(stock,start,end,API)
            data_proccess = loadDataApi.preprocess_data(data_file)
            # loadDataApi.printData(data_file)
            loadDataApi.save_data(save_file_path,data_proccess)
            data_frames[stock] = data_proccess.dropna().reset_index(drop=True)
        print("\t***   Data downloading finished    ***\n")
        return data_frames

    def load_5m_data(self, save=False):
        originals_path = 'datasets/5m/original/'
        clean_path = 'datasets/5m/'
        directory_content = listdir(originals_path)
        file_names = [join(originals_path, fn) for fn in directory_content]
        file_names = [fn for fn in file_names if isfile(fn)]
        data_frames = {}
        for fn in file_names:
            stock = fn.split()[0].split('/')[-1]
            df = pd.read_csv(fn, decimal=',')
            df['Date'] = pd.to_datetime(df['Date'].str.cat(df['Time'], sep=' '))
            df = df[['Date', 'Tick Range', 'Open', 'High', 'Low', 'Close']]
            df.iloc[:] = df.iloc[::-1].values
            float32_cols = ['Tick Range', 'Open', 'High', 'Low', 'Close']
            df[float32_cols] = df[float32_cols].astype(np.float32)
            data_frames[stock] = df
            if save:
                df.to_pickle(''.join((clean_path, stock, '_5m.pkl')))
        return data_frames

    def load_clean_5m_data(self):
        clean_path = 'datasets/5m/'
        directory_content = listdir(clean_path)
        file_names = [join(clean_path, fn) for fn in directory_content]
        file_names = [fn for fn in file_names if isfile(fn)]
        data_frames = {}
        print(file_names)
        for fn in file_names:
            stock = fn.split('_')[0].split('/')[-1]
            data_frames[stock] = pd.read_pickle(fn)
        return data_frames

def main():
    loadDataApi = LoadDataAPI()
    # loadDataApi.beginLoadData()
    dfs1 = loadDataApi.load_5m_data(save=True)
    # print(dfs1['AAPL'])
    # dfs2 = loadDataApi.load_clean_5m_data()
    # print(dfs2['AAPL'])
    # print(dfs2['AAPL']['Date'].iloc[0].day)

if __name__ == '__main__':
	main()
