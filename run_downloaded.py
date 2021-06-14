import numpy as np
from DCFcluster import DCFcluster
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings

from sklearn import cluster, datasets, metrics
import csv

Data = np.load("Dermatology.npy")
X = Data[:, range(X.shape[1] - 1)]
y = Data[:, X.shape[1] - 1]
# normalize dataset for easier parameter selection
result = DCFcluster.train(X, k = 20, beta = 0.4)
ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
with open("DCF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Dermatology",  k, beta, len(np.unique(result.labels)), ari,ami])

Data = np.load("Ecoli.npy")
X = Data[:, range(X.shape[1] - 1)]
y = Data[:, X.shape[1] - 1]
# normalize dataset for easier parameter selection
result = DCFcluster.train(X, k = 8, beta = 0.4)
ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
with open("DCF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Ecoli",  k, beta, len(np.unique(result.labels)), ari,ami])

Data = np.load("Glass.npy")
X = Data[:, range(X.shape[1] - 1)]
y = Data[:, X.shape[1] - 1]
# normalize dataset for easier parameter selection
result = DCFcluster.train(X, k = 16, beta = 0.4)
ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
with open("DCF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Glass",  k, beta, len(np.unique(result.labels)), ari,ami])

Data = np.load("Letter-Recognition.npy")
X = Data[:, range(X.shape[1] - 1)]
y = Data[:, X.shape[1] - 1]
# normalize dataset for easier parameter selection
result = DCFcluster.train(X, k = 20, beta = 0.4)
ami = metrics.adjusted_mutual_info_score(y.astype(int), result.labels.astype(int))
#ARI
ari = metrics.adjusted_rand_score(y.astype(int), result.labels.astype(int))
with open("DCF_Results.csv", 'a') as fd:
    writer = csv.writer(fd)
    writer.writerow(["DCF", "Letter-Recognition",  k, beta, len(np.unique(result.labels)), ari,ami])

