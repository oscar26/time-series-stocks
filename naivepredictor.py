# coding: UTF-8

import numpy as np
import pandas as pd
from dataformatter import DataFormatter

# Se usa Keras como la librería para manejar
# las redes neuronales.
from keras.models import Model
from keras.layers import Input, Dense

class NaivePredictor(object):
	"""docstring for NaivePredictor"""
	def __init__(self, dataset_path=None,	input_window_size=4, rolling_window_size=5,
			columns_to_standardize=None, columns_to_windowize=None, data=None):
		self.dataset_path = dataset_path
		self.input_window_size = input_window_size
		self.rolling_window_size = rolling_window_size
		self.columns_to_standardize = columns_to_standardize
		self.columns_to_windowize = columns_to_windowize
		self.data = data
		self.__preprocess_data()
		self.__create_model()

	def predict(self, point):
		"""Point debe ser un Data Frame de Pandas con las información
		necesaria para realizar la predicción."""
		# 1. Standardize point with training mean and standard deviation.
		# 2. Add it to the data.
		# 3. Windowize without Y (TODO: modify data formatter).
		# 4. Extract the last window.
		# 5. Make the prediction.
		# 6. Transform back the prediction.

	def fit_model(self, epochs=200, verbose=0):
		"""Entrenar el modelo para producción."""
		# Standardize inputs
		self.data, self.column_means, self.column_stds = self.__standardize_features(self.data, self.columns_to_standardize)
		# Windowize dataset
		fmt = DataFormatter()
		self.X, self.Y = fmt.windowize_series(self.data.as_matrix(), size=self.input_window_size, column_indexes=self.columns_to_windowize)
		self.model.fit(self.X, self.Y, epochs=epochs, batch_size=32, verbose=verbose)

	def test_model(self, n_splits=9, cv_runs=10, epochs=100, verbose=2):
		"""Evaluación del modelo usando validación cruzada
		hacia adelante."""
		from sklearn.model_selection import TimeSeriesSplit

		self.metrics = ['MSE', 'RMSE', 'MAE', 'MAPE']
		train_scores = np.zeros((cv_runs, n_splits, len(self.metrics)))
		test_scores = np.zeros((cv_runs, n_splits, len(self.metrics)))
		fmt = DataFormatter()
		tscv = TimeSeriesSplit(n_splits=n_splits)
		for j in xrange(cv_runs):
			#print('\nCross-validation run %i' % (j+1))
			i = 1
			for train_index, test_index in tscv.split(self.data['Close'].values):
				# División del conjunto de datos en entrenamiento y prueba
				train_df = self.data.loc[train_index]
				test_df = self.data.loc[test_index]
				# Estandarización del conjunto de datos
				train_data, training_means, training_stds = self.__standardize_features(train_df, self.columns_to_standardize)
				test_data = self.__standardize_features_for_test(test_df, self.columns_to_standardize, training_means, training_stds)
				# Extracción de ventanas de datos
				trainX, trainY = fmt.windowize_series(train_data.as_matrix(), size=self.input_window_size, column_indexes=self.columns_to_windowize)
				testX, testY = fmt.windowize_series(test_data.as_matrix(), size=self.input_window_size, column_indexes=self.columns_to_windowize)
				# Ajustando el modelo
				# print('Fold %i' % (i))
				self.model.fit(trainX, trainY, epochs=epochs, batch_size=32, validation_data=(testX, testY), verbose=verbose)
				# Evaluando cada partición de la validación cruzada hacia adelante
				train_score = self.model.evaluate(trainX, trainY, verbose=verbose)
				train_score = np.array([train_score[0], np.sqrt(train_score[0]), train_score[1], train_score[2]*100])
				test_score = self.model.evaluate(testX, testY, verbose=verbose)
				test_score = np.array([test_score[0], np.sqrt(test_score[0]), test_score[1], test_score[2]*100])
				# print('Train Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (train_score[0], train_score[1], train_score[2], train_score[3]))
				# print('Test Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE\n' % (test_score[0], test_score[1], test_score[2], test_score[3]))
				# [0: MSE, 1: RMSE, 2: MAE, 3: MAPE]
				train_scores[j, i-1, :] = train_score
				test_scores[j, i-1, :] = test_score
				i += 1
		self.train_results = train_scores.mean(axis=0).mean(axis=0)
		self.test_results = test_scores.mean(axis=0).mean(axis=0)
		# print(train_results)
		# print(test_results)

	def compile_model(self):
		metrics = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error']
		self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=metrics)

	def __create_model(self):
		n = self.input_window_size*len(self.columns_to_windowize) + self.data.shape[1] - len(self.columns_to_windowize)
		main_input = Input(shape=(n,), name='main_input', dtype='float32')
		x = Dense(n, activation='relu')(main_input)
		x = Dense(int(n*0.8), activation='relu')(x)
		x = Dense(int(n*0.6), activation='relu')(x)
		x = Dense(int(n*0.4), activation='relu')(x)
		x = Dense(int(n*0.2), activation='relu')(x)
		main_output = Dense(1, activation='linear', name='main_output')(x)
		self.model = Model(inputs=main_input, outputs=main_output)

	def __standardize_features(self, data, columns):
		"""Estandarización de los datos de entrenamiento.
		"""
		# Revisar si las columnas existen.
		if columns is None:
			raise('Passed column names is None.')
		if len(columns) == 0:
			raise('Passed column names is empty.')
		column_means = {}
		column_stds = {}
		for column in columns:
			mean = data.loc[:, column].mean()
			std = data.loc[:, column].std()
			data.loc[:, column] = (data.loc[:, column] - mean) / std
			column_means[column] = mean
			column_stds[column] = std
		return data, column_means, column_stds

	def __standardize_features_for_test(self, data, columns, training_means, training_stds):
		"""Estandarización de los datos de prueba usando la media
		y desviación estándar de los datos de entrenamiento.
		"""
		# Revisar si las columnas existen.
		if columns is None:
			raise('Passed column names is None.')
		if len(columns) == 0:
			raise('Passed column names is empty.')
		for column in columns:
			data.loc[:, (column)] = (data.loc[:, (column)] - training_means[column]) / training_stds[column]
		return data

	def __preprocess_data(self):
		"""Preprocesamiento del conjunto de datos."""
		if (self.data is None):
			df = pd.read_csv(self.dataset_path)
			#df['Close'] = df['Adj Close'] # DELETE
			date = pd.to_datetime(df['Date'])
			df.insert(0, 'Month', date.dt.month)
			df.insert(1, 'Day', date.dt.day)
			#df = df.drop('Adj Close', axis=1)
			#df = df.drop('Date', axis=1)
			new_column_order = ['Month', 'Day', 'Volume', 'Open', 'High', 'Low', 'Close']
			self.data = df.reindex(columns=new_column_order)
		else:
			pass
	def tester():
		"""Método exclusivo para pruebas locales de funcionamiento."""
		columns_to_standardize = ['Volume', 'Open', 'High', 'Low', 'Close']
		columns_to_windowize = [2, 3, 4, 5, 6]
		input_window_size = 5
		dataset_path = "datasets/AAPL.csv"
		predictor = NaivePredictor(dataset_path,
			columns_to_standardize=columns_to_standardize,
			columns_to_windowize=columns_to_windowize,
			input_window_size=input_window_size)
		predictor.compile_model()
		predictor.test_model(n_splits=9, epochs=90, verbose=0)

def test():
	naivePredictor=NaivePredictor()
	naivePredictor.tester()


if __name__ == '__main__':
	test()
