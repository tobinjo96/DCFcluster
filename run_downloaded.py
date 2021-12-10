import numpy as np
from DCFcluster import DCFcluster
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings

from sklearn.preprocessing import StandardScaler
from sklearn import cluster, datasets, metrics
import csv

Data = np.load("Dermatology.npy")
# normalize dataset for easier parameter selection
X = Data[:, range(Data.shape[1] - 1)]
y = Data[:, Data.shape[1] - 1]
nonans = np.isnan(X).sum(1) == 0
X = X[nonans,:]
y = y[nonans]
X = StandardScaler().fit_transform(X)
result = DCFcluster.train(X, k = 20, beta = 0.4)
ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
with open("DCF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Dermatology",  k, beta, len(np.unique(result.labels)), ari,ami])

Data = np.load("Ecoli.npy")
# normalize dataset for easier parameter selection
X = Data[:, range(Data.shape[1] - 1)]
y = Data[:, Data.shape[1] - 1]
nonans = np.isnan(X).sum(1) == 0
X = X[nonans,:]
y = y[nonans]
X = StandardScaler().fit_transform(X)
result = DCFcluster.train(X, k = 8, beta = 0.4)
ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
with open("DCF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Ecoli",  k, beta, len(np.unique(result.labels)), ari,ami])

Data = np.load("Glass.npy")
# normalize dataset for easier parameter selection
X = Data[:, range(Data.shape[1] - 1)]
y = Data[:, Data.shape[1] - 1]
nonans = np.isnan(X).sum(1) == 0
X = X[nonans,:]
y = y[nonans]
X = StandardScaler().fit_transform(X)
result = DCFcluster.train(X, k = 16, beta = 0.4)
ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
with open("DCF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Glass",  k, beta, len(np.unique(result.labels)), ari,ami])

Data = np.load("Letter-Recognition.npy")
# normalize dataset for easier parameter selection
X = Data[:, range(Data.shape[1] - 1)]
y = Data[:, Data.shape[1] - 1]
nonans = np.isnan(X).sum(1) == 0
X = X[nonans,:]
y = y[nonans]
X = StandardScaler().fit_transform(X)
result = DCFcluster.train(X, k = 20, beta = 0.4)
ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
with open("DCF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Letter-Recognition",  k, beta, len(np.unique(result.labels)), ari,ami])

