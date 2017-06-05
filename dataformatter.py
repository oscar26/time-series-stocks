import numpy as np
import pandas as pd
from os.path import isfile

class DataFormatter(object):
	def get_1_and_15_min(self, file_path, column_names=None):
		'''Data must be 1 minute tick bar.
		'''
		if not file_path or file_path == '':
			raise('File path is none or empty.\n')
		if not isfile(file_path):
			raise('The file path is not a file.\n')
		df = pd.read_csv(file_path, names=column_names)
		num_15_min_frames = df.shape[0] / 15
		indexes = np.arange(num_15_min_frames) * 15
		return df, df.iloc[indexes]

	def windowize_series(self, data, size=10, horizon=1, column_indexes=None):
		'''Last column of the dataframe must be the target Y.
		'''
		if type(data).__module__ != np.__name__:
			raise('The data must be a numpy array.')
		if data is None or data.size == 0:
			raise('The array is none or empty.')
		num_windows = data.shape[0] - size - horizon + 1
		if column_indexes is None:
			input_vector_length = data.shape[1] * size
		else:
			input_vector_length = size*len(column_indexes) + data.shape[1] - len(column_indexes)
			all_indexes = xrange(data.shape[1])
			ignored_indexes = [index for index in all_indexes if index not in column_indexes]
		X = np.zeros((num_windows, input_vector_length))
		Y = data[size+horizon-1:, data.shape[1]-1]
		for i in xrange(num_windows):
			if column_indexes is None:
				X[i, :] = np.reshape(data[i:i+size, :], size*data.shape[1], order='F')
			else:
				input_vector = np.reshape(data[i:i+size, column_indexes], input_vector_length - len(ignored_indexes), order='F')
				input_vector = np.insert(input_vector, 0, data[i, ignored_indexes])
				X[i, :] = input_vector
		return X, Y

	def get_last_window(self, data, size=10, horizon=1, column_indexes=None):
		num_windows = data.shape[0] - size - horizon + 1
		if column_indexes is None:
			last_window = np.reshape(data[-size:, :], size*data.shape[1], order='F')
		else:
			input_vector_length = size*len(column_indexes) + data.shape[1] - len(column_indexes)
			all_indexes = xrange(data.shape[1])
			ignored_indexes = [index for index in all_indexes if index not in column_indexes]
			last_window = np.reshape(data[-size:, column_indexes], input_vector_length - len(ignored_indexes), order='F')
			last_window = np.insert(last_window, 0, data[-size, ignored_indexes])
		return last_window

def main():
	file_path = 'datasets/dummyData.csv'
	data_fmt = DataFormatter()
	frames_1_min, frames_15_min = data_fmt.get_1_and_15_min(file_path, ['Hour', 'Minute', 'Close'])
	# print frames_15_min

	data_fmt.windowize_series(frames_15_min.as_matrix())

if __name__ == '__main__':
	main()