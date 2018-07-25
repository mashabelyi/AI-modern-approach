import sys
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

class PerceptronLearn:
	def __init__(self, df, outFile):
		self.df = df
		self.w = [0, 0, 0] # [w1, w2, b]
		self.converged = False
		self.steps = 100
		self.outFile = outFile

	def learn(self):
		f = open(self.outFile, 'w')

		while not self.converged and self.steps > 0:
			# print("---{}---".format(100-self.steps))
			self.converged = True
			for index, row in df.iterrows():
				y = row['l']
				fx = np.dot(self.w, [row["x1"], row["x2"], row["x0"]])
				if y*fx <=0:
					self.converged = False
					self.w = np.sum([self.w, [y*row["x1"], y*row["x2"], y*row["x0"]]], axis=0)
					f.write("{},{},{}\n".format(int(self.w[0]), int(self.w[1]), int(self.w[2])))
					# print(self.w)

			
			self.steps -= 1
		f.close()
		return self.w

	def plot(self):
		fig, ax = plt.subplots()
		# scatter data
		ax.scatter(self.df['x1'], self.df['x2'], c=self.df['l'])
		# plot line
		x = self.df["x1"]
		y = (-self.w[2]-self.w[0]*x) / self.w[1]
		ax.plot(x,y)

		plt.show()



if __name__ == '__main__':

	fin = sys.argv[1]
	fout =sys.argv[2]

	df = read_csv(fin, names=['x1', 'x2', 'l'])
	# add bias term
	df.insert(loc=0, column='x0', value=np.ones(len(df.x1)))

	pla = PerceptronLearn(df, fout)
	pla.learn()
	pla.plot()

