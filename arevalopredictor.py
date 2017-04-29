import numpy as np
import pandas as pd
from dataformatter import DataFormatter
# import pylab as plt
from keras.models import Model
from keras.layers import Input, Dense

class ArevaloPredictor(object):
	def __init__(
			self, dataset_path,	input_window_size=4, rolling_window_size=5,
			columns_to_standardize=None, columns_to_windowize=None):
		self.dataset_path = dataset_path
		self.input_window_size = input_window_size
		self.rolling_window_size = rolling_window_size
		self.columns_to_standardize = columns_to_standardize
		self.columns_to_windowize = columns_to_windowize
		self.__preprocess_data()
		self.__create_model()

	def test_model(self, epochs=200, verbose=2):
		from sklearn.model_selection import TimeSeriesSplit

		fmt = DataFormatter()
		tscv = TimeSeriesSplit(n_splits=9)
		i = 1
		for train_index, test_index in tscv.split(self.data['plr'].values):
			# Splitting dataset
			train_df = self.data.iloc[train_index]
			test_df = self.data.iloc[test_index]
			# Normalizing dataset
			train_data, training_means, training_stds = self.__standardize_features(train_df, self.columns_to_standardize)
			test_data = self.__standardize_features_for_test(test_df, self.columns_to_standardize, training_means, training_stds)
			# Creating windowed dataset
			trainX, trainY = fmt.windowize_series(train_data.as_matrix(), size=self.input_window_size, column_indexes=self.columns_to_windowize)
			testX, testY = fmt.windowize_series(test_data.as_matrix(), size=self.input_window_size, column_indexes=self.columns_to_windowize)
			# Fitting the model
			print('\nFold %i\n' % (i))
			self.model.fit(trainX, trainY, epochs=epochs, batch_size=32, validation_data=(testX, testY), verbose=verbose)
			# Evaluating each fold
			trainScore = self.model.evaluate(trainX, trainY, verbose=verbose)
			print('\nTrain Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (trainScore[0], np.sqrt(trainScore[0]), trainScore[1], trainScore[2]*100))
			testScore = self.model.evaluate(testX, testY, verbose=verbose)
			print('\nTest Score: %.5f MSE, %.5f RMSE, %.5f MAE, %.5f%% MAPE' % (testScore[0], np.sqrt(testScore[0]), testScore[1], testScore[2]*100))
			i += 1

	def fit_model(self, epochs=200, verbose=2):
		'''Method to be used in production.
		'''
		# Standardize inputs
		self.data, self.column_means, self.column_stds = self.__standardize_features(self.data, self.columns_to_standardize)
		self.data = self.data.as_matrix()
		# Windowize dataset
		fmt = DataFormatter()
		self.X, self.Y = fmt.windowize_series(self.data.as_matrix(), size=self.input_window_size, column_indexes=self.columns_to_windowize)
		self.model.fit(self.X, self.Y, epochs=epochs, batch_size=32, verbose=verbose)

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

	def __preprocess_data(self):
		# Data loading and formatting
		fmt = DataFormatter()
		df_1_min, df_15_min = fmt.get_1_and_15_min(self.dataset_path, ['Hour', 'Minute', 'Close'])
		# Computing standard deviation
		df_15_min = pd.concat([df_15_min, self.__compute_std(df_1_min)], axis=1)
		# Computing trends
		df_15_min = pd.concat([df_15_min, self.__compute_trend(df_1_min)], axis=1)
		# Computing pseudo-log-return
		df_15_min = pd.concat([df_15_min, self.__compute_pseudo_log_return(df_1_min)], axis=1)
		# Converting to Numpy array in order to windowize it and dropping
		# the 'Close' column as it is not needed for training or inference
		# and the first row as it contains NaN values.
		self.data = df_15_min.drop('Close', axis=1).drop(df_15_min.index[0])

	def __standardize_features(self, data, columns):
		# Check if column names exist.
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
		# Check if column names exist.
		if columns is None:
			raise('Passed column names is None.')
		if len(columns) == 0:
			raise('Passed column names is empty.')
		for column in columns:
			data.loc[:, (column)] = (data.loc[:, (column)] - training_means[column]) / training_stds[column]
		return data

	def __compute_pseudo_log_return(self, df_1_min):
		'''The pseudo-log-return is computed from 1-minute ticks and
			 resampled to 15-minute ticks. Perhaps, this approach could
			 be flawed.
		'''
		if not isinstance(df_1_min, pd.DataFrame):
			raise('The object container is not a pandas data frame.')
		if df_1_min.empty:
			raise('The data frame is empty.')
		# Computing rolling average
		df = df_1_min['Close'].rolling(window=self.rolling_window_size).mean().to_frame()
		# Computing returns
		df = np.log(df) - np.log(df.shift(1))
		# Resampling to 15 minutes
		num_15_min_frames = df_1_min.shape[0] / 15
		indexes = np.arange(num_15_min_frames) * 15
		df.rename(columns={'Close': 'plr'}, inplace=True)
		return df.iloc[indexes]

	def __compute_std(self, df_1_min):
		'''The standar deviation is computed from 1-minute ticks and
			 resampled to 15-minute ticks. Perhaps, this approach could
			 be flawed.
		'''
		if not isinstance(df_1_min, pd.DataFrame):
			raise('The object container is not a pandas data frame.')
		if df_1_min.empty:
			raise('The data frame is empty.')
		# Computing rolling standard deviation
		df = df_1_min['Close'].rolling(window=self.rolling_window_size).std().to_frame()
		# Resampling to 15 minutes
		num_15_min_frames = df_1_min.shape[0] / 15
		indexes = np.arange(num_15_min_frames) * 15
		df.rename(columns={'Close': 'std'}, inplace=True)
		return df.iloc[indexes]

	def __compute_trend(self, df_1_min):
		if not isinstance(df_1_min, pd.DataFrame):
			raise('The object container is not a pandas data frame.')
		if df_1_min.empty:
			raise('The data frame is empty.')
		# Computing the trends (fitting a line)
		num_15_min_frames = df_1_min.shape[0] / 15
		trends = pd.DataFrame(np.zeros((num_15_min_frames, 1)), columns=['trend'])
		trends['trend'][0] = np.nan
		trends.set_index(np.arange(num_15_min_frames)*15, inplace=True)
		x = np.arange(15)
		y = df_1_min['Close'].as_matrix()
		A = np.vstack([x, np.ones(len(x))]).T
		for i in xrange(num_15_min_frames):
			start_pos = i * 15
			end_pos = start_pos + 15
			m, c = np.linalg.lstsq(A, y[start_pos:end_pos])[0]
			trends['trend'][(i+1)*15] = m
		return trends

def main():
	columns_to_standardize = ['std', 'trend', 'plr']
	columns_to_windowize = [2, 3, 4]
	predictor = ArevaloPredictor(
		'datasets/dummyData.csv',	columns_to_standardize=columns_to_standardize,
		columns_to_windowize=columns_to_windowize)
	predictor.compile_model()
	predictor.test_model(epochs=200)

if __name__ == '__main__':
	main()