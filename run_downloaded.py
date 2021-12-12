import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, './src')

import numpy as np
from DCFcluster import DCFcluster
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import csv

if __name__ == "__main__":
  #==========================================================================================================================================================================
  Data = np.load("Dermatology.npy")
  X = Data[:, range(Data.shape[1] - 1)]
  y = Data[:, Data.shape[1] - 1]
  #Remove NaNs and Standardize
  nonans = np.isnan(X).sum(1) == 0
  X = X[nonans,:]
  y = y[nonans]
  X = StandardScaler().fit_transform(X)
  #Set Parameter Values
  k = 13
  beta = 0.4
  result = DCFcluster.train(X, k = k, beta = beta)
  #Compute Adjusted Mutual Information & Adjusted Rand Index
  ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
  ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
  with open("DCF_Results.csv", 'a') as fd:
      writer = csv.writer(fd)
      writer.writerow(["DCF", "Dermatology",  k, beta,  ari,ami])
      fd.close()
  
  #==========================================================================================================================================================================
  Data = np.load("Ecoli.npy")
  X = Data[:, range(Data.shape[1] - 1)]
  y = Data[:, Data.shape[1] - 1]
  #Remove NaNs and Standardize
  nonans = np.isnan(X).sum(1) == 0
  X = X[nonans,:]
  y = y[nonans]
  X = StandardScaler().fit_transform(X)
  #Set Parameter Values
  k = 14
  beta = 0.4
  result = DCFcluster.train(X, k = k, beta = beta)
  #Compute Adjusted Mutual Information & Adjusted Rand Index
  ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
  ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
  with open("DCF_Results.csv", 'a') as fd:
      writer = csv.writer(fd)
      writer.writerow(["DCF", "Ecoli",  k, beta, ari,ami])
      fd.close()
  
  #==========================================================================================================================================================================
  Data = np.load("Glass.npy")
  X = Data[:, range(Data.shape[1] - 1)]
  y = Data[:, Data.shape[1] - 1]
  #Remove NaNs and Standardize
  nonans = np.isnan(X).sum(1) == 0
  X = X[nonans,:]
  y = y[nonans]
  X = StandardScaler().fit_transform(X)
  #Set Parameter Values
  k = 15
  beta = 0.4
  result = DCFcluster.train(X, k = k, beta = beta)
  #Compute Adjusted Mutual Information & Adjusted Rand Index
  ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
  ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
  with open("DCF_Results.csv", 'a') as fd:
      writer = csv.writer(fd)
      writer.writerow(["DCF", "Glass",  k, beta,  ari,ami])
      fd.close()
  
  #==========================================================================================================================================================================
  Data = np.load("Letter-Recognition.npy")
  X = Data[:, range(Data.shape[1] - 1)]
  y = Data[:, Data.shape[1] - 1]
  #Remove NaNs and Standardize
  nonans = np.isnan(X).sum(1) == 0
  X = X[nonans,:]
  y = y[nonans]
  X = StandardScaler().fit_transform(X)
  #Set Parameter Values
  k = 21
  beta = 0.4
  result = DCFcluster.train(X, k = k, beta = beta)
  #Compute Adjusted Mutual Information & Adjusted Rand Index
  ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
  ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
  with open("DCF_Results.csv", 'a') as fd:
      writer = csv.writer(fd)
      writer.writerow(["DCF", "Letter-Recognition",  k, beta, ari,ami])
      fd.close()

