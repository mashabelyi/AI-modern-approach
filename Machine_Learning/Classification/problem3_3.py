import sys
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree

def classify(type, clf, X_train, y_train, X_test, y_test):
	print("##########################")
	print("{}".format(type))
	print("##########################")


	# fit
	clf.fit(X_train, y_train)

	print("Best parameters set found on trainig set:")
	print(clf.best_params_)
	best_score = clf.best_score_

	# Test on test set
	y_true, y_pred = y_test, clf.predict(X_test)
	# report = classification_report(y_true, y_pred)
	# print(report)
	test_score = accuracy_score(y_true, y_pred)

	print("{},{},{}\n".format(type, best_score, test_score))

def make_meshgrid(x, y, h=.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


if __name__ == '__main__':

	fin = sys.argv[1]
	fout = sys.argv[2]

	df = read_csv(fin)

	# Split data into test, train sets
	X_train, X_test, y_train, y_test = train_test_split(df[['A','B']].values, df['label'].values, test_size=0.4, stratify=df['label'].values)

	# Visualize test, train sets
	# fig, ax = plt.subplots()
	# ax.scatter(X_train[:,0], X_train[:,1], c=y_train)
	# plt.title('Training Set')
	# plt.show()

	# fig, ax = plt.subplots()
	# ax.scatter(X_test[:,0], X_test[:,1], c=y_test)
	# plt.title('Test Set')
	# plt.show()
	
	models = []
	#################################
	# SVM with Linear Kernel
	#################################
	param_grid = [
		{'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'kernel': ['linear']}
	]
	clf = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy')
	models.append(clf)
	# classify("svm_linear", clf, X_train, y_train, X_test, y_test)

	#################################
	# SVM with Polynomial Kernel
	#################################
	param_grid = [
		{'C': [0.1, 1, 3], 'kernel': ['poly'], 'degree':[4,5,6], 'gamma': [0.1,0.5]}
	]

	clf = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy')
	models.append(clf)
	# classify("svm_polynomial", clf, X_train, y_train, X_test, y_test)


	#################################
	# SVM with RBF Kernel
	#################################
	param_grid = [
		{'C': [0.1, 0.5, 1, 5, 10, 50, 100], 'gamma': [0.1, 0.5, 1, 3, 6, 10], 'kernel': ['rbf']}
	]
	clf = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy')
	models.append(clf)
	# classify("svm_rbf", clf, X_train, y_train, X_test, y_test)


	########################
	# LOGISTIC REGRESSION
	########################
	param_grid = [
		{'C': [0.1, 0.5, 1, 5, 10, 50, 100]}
	]

	clf = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='accuracy')
	models.append(clf)
	# classify("logistic", clf, X_train, y_train, X_test, y_test)

	########################
	# k-Nearest Neighbors
	########################
	param_grid = [
		{'n_neighbors': np.arange(1,51,1), 'leaf_size': np.arange(5,65,5)}
	]
	clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
	models.append(clf)
	# classify("knn", clf, X_train, y_train, X_test, y_test)

	########################
	# Decision Tree
	########################
	param_grid = [
		{'max_depth': np.arange(1,51,1), 'min_samples_split': np.arange(2,11,1)}
	]
	
	clf = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=5, scoring='accuracy')
	models.append(clf)
	# classify("decision_tree", clf, X_train, y_train, X_test, y_test)


	########################
	# Random Forest
	########################
	param_grid = [
		{'max_depth': np.arange(1,51,1), 'min_samples_split': np.arange(2,11,1)}
	]
	
	clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
	models.append(clf)
	# classify("random_forest", clf, X_train, y_train, X_test, y_test)

	########################
	# FIT!
	########################

	models = (clf.fit(X_train, y_train) for clf in models)

	# title for the plots
	titles = ('SVM with Linear Kernel',
	          'SVM with Polynomial Kernel',
	          'SVM with RBF Kernel',
	          'Logistic Regression',
	          'k-Nearest Neighbors',
	          'Decision Trees',
	          'Random Forest')

	# Set-up 2x2 grid for plotting.
	fig, sub = plt.subplots(2, 4)
	plt.subplots_adjust(wspace=0.4, hspace=0.4)

	X0, X1 = X_train[:, 0], X_train[:, 1]
	xx, yy = make_meshgrid(X0, X1)

	f = open(fout, 'w')
	for clf, title, ax in zip(models, titles, sub.flatten()):

		print(title)
		print(clf.best_params_)
		best_score = clf.best_score_
		y_true, y_pred = y_test, clf.predict(X_test)
		test_score = accuracy_score(y_true, y_pred)

		print("{},{},{}\n".format(title, best_score, test_score))
		f.write("{},{},{}\n".format(title, best_score, test_score))


		plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
		ax.scatter(X0, X1, c=y_train, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
		ax.set_xlim(xx.min(), xx.max())
		ax.set_ylim(yy.min(), yy.max())
		ax.set_xlabel('x1')
		ax.set_ylabel('x2')
		ax.set_xticks(())
		ax.set_yticks(())
		ax.set_title(title)

	f.close()
	plt.show()



	