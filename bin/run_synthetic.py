import numpy as np
from DCFcluster import DCFcluster
from itertools import cycle, islice
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import warnings

from sklearn import cluster, datasets, metrics

np.random.seed(0)
n_samples = 1500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)
                        
         

f = plt.figure(figsize=(16.5 , 3))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
                
plot_num = 1
dataset_list = [noisy_circles, noisy_moons, varied, aniso, blobs, no_structure]

for i_dataset,  dataset in enumerate(dataset_list):
    Data = dataset
    X = Data[0]
    y = Data[1]
    # normalize dataset for easier parameter selection
    result = DCFcluster.train(X, k = 40, beta = 0.4)
    plt.subplot(1, len(dataset_list), plot_num)
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                         '#f781bf', '#a65628', '#984ea3',
                                         '#999999', '#e41a1c', '#dede00']),
                                  int(max(result.labels) + 1))))
    
    plt.xlim(X[:,0].min(), X[:,0].max())
    plt.ylim(X[:,1].min(), X[:,1].max())
    plt.scatter(X[:, 0], X[:, 1], s = 1, color = colors[result.labels], marker = "o")
    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('k= {0} beta= {1}'.format(str(40), str(0.7))).lstrip('0'),
             transform=plt.gca().transAxes, size=5,
             horizontalalignment='right')
    plot_num += 1

plt.show()
f.savefig("DCF_Synthetic.pdf", bbox_inches='tight')
plt.close()
