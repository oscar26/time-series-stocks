import numpy as np
import pylab as plt
from numpy.random import normal

class DummyDataGenerator(object):
	def __init__(self, file_name="dummyData", folder_path='datasets/', return_mean=0.000001, return_sd=0.0002):
		self.file_name = file_name
		self.folder_path = folder_path
		self.return_mean = return_mean
		self.return_sd = return_sd
		self.market_day_minutes = 390

	def generate_data(self, num_days=251, starting_price=100):
		num_datapoints = num_days * self.market_day_minutes
		self.dataset = np.zeros((num_datapoints, 3))
		hour = minute = 0
		self.dataset[0, :] = (hour, minute, starting_price)
		for i in xrange(1, num_datapoints):
			if i % 390 == 0:
				price = self.dataset[i-1, 2] + self.dataset[i-1, 2]*normal(0, self.return_sd*10)
				hour = minute = 0
			else:
				price = self.dataset[i-1, 2] + self.dataset[i-1, 2]*normal(self.return_mean, self.return_sd)
				minute = (minute + 1) % 60
				if minute == 0:
					hour += 1
			self.dataset[i, :] = (hour, minute, price)

	def plot_data(self):
		plt.plot(self.dataset[:, 2])
		plt.show()

	def save_data(self):
		if self.folder_path[-1] != '/':
			raise('Folder path must end with \'/\'')
		if raw_input('Do you want to save the generated data? (y/n): ') == 'y':
			save_location = self.folder_path + self.file_name + '.csv'
			np.savetxt(save_location, self.dataset, delimiter=',')
			print 'File successfully saved.'
		else:
			print 'File not saved.'

def main():
	data_generator = DummyDataGenerator()
	data_generator.generate_data()
	data_generator.plot_data()
	data_generator.save_data()

if __name__ == '__main__':
	main()