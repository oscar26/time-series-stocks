# coding: UTF-8

import numpy as np
import pandas as pd
from dataformatter import DataFormatter

# Se usa Keras como la librería para manejar
# las redes neuronales.
from keras.models import Model
from keras.layers import Input, Dense

class TransformPredictor(object):
	"""docstring for TransformPredictor"""
	def __init__(self, dataset_path=None,	input_window_size=4,
			columns_to_standardize=None, columns_to_windowize=None, data=None):
		self.dataset_path = dataset_path
		self.input_window_size = input_window_size
		self.columns_to_standardize = columns_to_standardize
		self.columns_to_windowize = columns_to_windowize
		self.data = data
		self.__preprocess_data()
		self.__create_model()

	def predict(self, point, last_price):
		"""Point debe ser un Data Frame de Pandas con las información
		necesaria para realizar la predicción."""
		# 1. Standardize point with training mean and standard deviation.
		test_data = self.__standardize_features_for_test(point, self.columns_to_standardize, self.column_means, self.column_stds)
		# 2. Add it to the data.
		df = pd.concat([self.data, test_data])
		# 3. Windowize.
		fmt = DataFormatter()
		X, Y = fmt.windowize_series(df.as_matrix(), size=self.input_window_size, column_indexes=self.columns_to_windowize)
		# 4. Extract the last window.
		last_window = fmt.get_last_window(df.as_matrix(), size=self.input_window_size, column_indexes=self.columns_to_windowize)
		# 5. Compute the error.
		train_score = self.model.evaluate(X, Y, verbose=0)
		train_score = np.array([train_score[0], np.sqrt(train_score[0]), train_score[1], train_score[2]*100])
		# 6. Make the prediction.
		prediction = self.model.predict(last_window)
		# 7. Computing prediction intervals
		pred_upper = prediction + 1.96 * train_score[1]
		pred_lower = prediction - 1.96 * train_score[1]
		# 8. Transform back the prediction.
		prediction = last_price * np.exp(prediction)
		pred_upper = last_price * np.exp(pred_upper)
		pred_lower = last_price * np.exp(pred_lower)
		return prediction, [pred_lower, pred_upper]

	def fit_model(self, epochs=200, verbose=0):
		"""Entrenar el modelo para producción."""
		# Patching
		self.column_means = {}
		self.column_stds = {}
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
			# print('\nCross-validation run %i' % (j+1))
			i = 1
			for train_index, test_index in tscv.split(self.data['LogReturn'].values):
				# División del conjunto de datos en entrenamiento y prueba
				train_df = self.data.loc[train_index]
				test_df = self.data.loc[test_index]
				# Estandarización del conjunto de datos
				if len(self.columns_to_standardize) != 0:
					train_data, training_means, training_stds = self.__standardize_features(train_df, self.columns_to_standardize)
					test_data = self.__standardize_features_for_test(test_df, self.columns_to_standardize, training_means, training_stds)
				else:
					train_data = train_df
					test_data = test_df
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
		# x = Dense(int(n*0.5), activation='relu')(x)
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
			df['Close'] = df['Adj Close'] # DELETE
			date = pd.to_datetime(df['Date'])
			df.insert(0, 'Month', date.dt.month)
			df.insert(1, 'Day', date.dt.day)
			self.data = df
		if 'Adj Close' in self.data.columns:
			self.data.drop('Adj Close', axis=1, inplace=True)
		if 'Date' in self.data.columns:
			self.data.drop('Date', axis=1, inplace=True)
		if 'Volume' in self.data.columns:
			self.data.drop('Volume', axis=1, inplace=True)
		if 'Open' in self.data.columns:
			self.data.drop('Open', axis=1, inplace=True)
		if 'High' in self.data.columns:
			self.data.drop('High', axis=1, inplace=True)
		if 'Low' in self.data.columns:
			self.data.drop('Low', axis=1, inplace=True)
		if 'Month' in self.data.columns and 'Day' in self.data.columns:
			self.data = self.data.reindex(columns=['Month', 'Day', 'Close'])
		# Transformación de no estacionario a estacionario
		self.data.insert(self.data.shape[1], 'LogReturn', self.__compute_log_return(self.data['Close']))
		self.data.drop('Close', axis=1, inplace=True)
		# Eliminación de la primera fila por contener un valor inválido
		self.data.drop(self.data.index[0], inplace=True)
		self.data.reset_index(inplace=True)

	def __compute_log_return(self, prices):
		if prices.empty:
			raise Exception('The data frame is empty.')
		return np.log(prices) - np.log(prices.shift(1))

def test():
	"""Método exclusivo para pruebas locales de funcionamiento."""
	# columns_to_standardize = ['LogReturn']
	columns_to_standardize = []
	columns_to_windowize = [2]
	input_window_size = 20
	dataset_path = "datasets/AAPL.csv"
	predictor = TransformPredictor(dataset_path,
		columns_to_standardize=columns_to_standardize,
		columns_to_windowize=columns_to_windowize,
		input_window_size=input_window_size)
	predictor.compile_model()
	predictor.test_model(n_splits=9, epochs=100, verbose=0)

if __name__ == '__main__':
	test()