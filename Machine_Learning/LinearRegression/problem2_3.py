import sys
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class LinearRegression:
	def __init__(self, data, alpha, iter):
		self.x = data[:,:3]
		self.y = data[:,3]
		self.w = np.zeros(len(data[0])-1)

		self.alpha = alpha
		self.iter = iter
		self.n = len(data)
		# print(len(data))
		# print(len(data[0]))
		


	def learn(self):
		while self.iter > 0:
			fx = self.x.dot(self.w)

			# update weights
			self.w -= (self.alpha/self.n) * np.transpose(self.x).dot(fx-self.y)
			self.iter -= 1

			# error = np.sum(np.square(fx)) / self.n
			# print(error)

		return self.w

	def plot(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(self.x[:,1], self.x[:,2], self.y[:])
		ax.set_xlabel('age', fontsize=15)
		ax.set_ylabel('weight', fontsize=15)
		ax.set_zlabel('height', fontsize=15)
		# ax.hold(True)

		xx, yy = np.meshgrid(range(-5,5), range(-5,5))
		ii = np.ones(100)
		pts = np.transpose([ii, xx[:].reshape(100), yy[:].reshape(100)])
		# print(pts)
		zz = pts.dot(self.w).reshape(10,10)
		# print(zz)
		ax.plot_surface(xx,yy,zz, color='green')

		plt.show()



if __name__ == '__main__':

	fin = sys.argv[1]
	fout = sys.argv[2]

	df = read_csv(fin, names=['age', 'weight', 'height'])
	df.insert(loc=0, column='b', value=np.ones(len(df.age)))

	# Normalize
	df['age_norm'] =  (df['age'] - df['age'].mean()) /  df['age'].std()
	df['weight_norm'] =  (df['weight'] - df['weight'].mean()) /  df['weight'].std()
	# print(df.head())	

	

	data = df[['b', 'age_norm', 'weight_norm', 'height']].values
	# print(data)
	# print(np.transpose(data))
	# print(data[:,3])

	learningRates =[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
	f = open(fout, 'w')

	for r in learningRates:
		n_iter = 100
		LR = LinearRegression(data, r, n_iter)
		w = LR.learn()
		
		# print("{},{},{},{},{}".format(r, n_iter, w[0], w[1], w[2]))
		f.write("{},{},{},{},{}\n".format(r, n_iter, w[0], w[1], w[2]))

	
	# Choose your own alpha and n+iterations
	rate = 0.1
	niter = 1000
	LR = LinearRegression(data, rate, niter)
	w = LR.learn()
	f.write("{},{},{},{},{}\n".format(rate, niter, w[0], w[1], w[2]))
	f.close()

	# Visualize fit
	LR.plot()
	


	