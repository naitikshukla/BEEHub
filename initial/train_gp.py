# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 15:04:19 2017

@author: naitikshukla
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import explained_variance_score,mean_squared_error
from sklearn.gaussian_process.kernels import RBF#,Matern,RationalQuadratic
from sklearn.model_selection import RepeatedKFold
from matplotlib import pyplot as plt
import numpy as np
import pickle
from sklearn.externals import joblib
import csv,time

if __name__ == '__main__':
	#instantiate list to append
	stime = time.time()
	yc = []
	xc = []
	tc = []
	p = 3
	q = 3
	with open('beehub/testdataset/DATACOMP.csv') as csvfile:		#Open Simulation file
		for x in csv.reader(csvfile,delimiter=','):		#render for all rows
			if x ==[]: continue							#Check for null rows and skip
			elif x[0]=='yc': continue					#check for header row and skip
			else:
				yc.append(float(x[0]))					#extract yc from 1st location, also convert to float
				xc.append(x[1:p+1])						#extract xc from 2,3 and 4th location
				tc.append(x[p+1:p+q+1])						#extract tc from 4,6 and 7th location

	n = len(yc)										#check number of rows extracted

	xc = [[float(s) for s in sublist] for sublist in xc]	#convert xc items to float
	tc = [[float(s) for s in sublist] for sublist in tc] 	#convert tc items to float

	x = np.r_['1',xc,tc]
	x = np.reshape(x,(n, p+q)) #simulation input			#reshape the list to numpy array
	y = np.reshape(yc,(n, 1))	#simulation otput			#reshape the list to numpy array

	#standardization of y
	y_sd = np.std(y, dtype=np.float32)
	y_mu = np.mean(y, dtype=np.float32)
	y =  (y - y_mu) / y_sd

	#normalization of x
	#Put design points x on [0,1]
	for i in range(x.shape[1]):
		x_min = min(x[:,i])
		x_max = max(x[:,i])
		x[:,i] = (x[:,i] - x_min) / (x_max - x_min)

	del tc,x_max,x_min,xc,y_mu,y_sd,yc,i

	#initialize Gaussian instant
	#uncomment the kernel want to use
	#kernel = Matern()
	#kernel = ConstantKernel() + Matern(length_scale=2, nu=3/2) + WhiteKernel(noise_level=1)  #need to check
	kernel = RBF()
	#kernel = RationalQuadratic()
	random_state = 12883823

	gp = GaussianProcessRegressor(kernel=kernel, alpha=0,optimizer=None, normalize_y=True,n_restarts_optimizer=10,random_state=random_state)
	'''
	alpha is noise ,Value added to the diagonal of the kernel matrix during fitting.
	 Larger values correspond to increased noise level in the observations
	In our previous case 1/lambda_eta was diagonal element, which was gamma(10,10)
	s = np.random.gamma(shape, scale, 1000)
	'''

	#MSE = []
	#variance = []
	rkf = RepeatedKFold(n_splits=2, n_repeats=1, random_state=random_state)
	mx = 0
	scores = list()
	model_num =1
	for train, test in rkf.split(x,y):

		y_fit = gp.fit(x[train], y[train])
		y_pred, y_std = y_fit.predict(x[test], return_std=True)
		scores.append(y_fit.score(x[test], y[test]))

		mse= mean_squared_error(y[test], y_pred)
		var = explained_variance_score(y[test], y_pred)
		#print("MSE is:",mse,"Variance is: ",var)
		#MSE.append(mse)
		#variance.append(var)
		filename = 'finalized_model_gp.pkl'
		if mx==0:													#check for first loop
			mx = abs(var-mse)
			# save the model to disk
			joblib.dump(y_fit, filename)
			#pickle.dump(y_fit, open(filename, 'wb'))#save the model
			print("saving model: ",model_num," with MSE:",mse," Variance:",var)
		elif abs(var-mse)>mx:
			mx =abs(var-mse)
			joblib.dump(y_fit, filename)
			#pickle.dump(y_fit, open(filename, 'wb'))
			print("Now saving model:",model_num," with MSE:",mse," Variance:",var)		#if any other greater model
		model_num += 1
	print('average score is:',np.mean(scores))
	#	 average score is: 0.999370844046   #
	print("Total Time for GPR fitting: %.3f" % (time.time() - stime))
