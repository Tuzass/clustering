import time
import warnings
from itertools import cycle, islice
import numpy as np
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler

# changing seed and random_state will result in different datasets
# the final datasets used had seed = 30 and random_state = 170
seed = 30
random_state = 170

n_samples = 500
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

default_base = {
    "quantile": 0.3,
    "eps": 0.3,
    "damping": 0.9,
    "preference": -200,
    "n_neighbors": 3,
    "n_clusters": 3,
    "min_samples": 7,
    "xi": 0.05,
    "min_cluster_size": 0.1,
    "allow_single_cluster": True,
    "hdbscan_min_cluster_size": 15,
    "hdbscan_min_samples": 3,
    "random_state": 42,
}

datasets = [
    (
        datasets.make_circles(n_samples=n_samples, factor=0.1, noise=0.05, random_state=seed),
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        },
    ),
    (
        datasets.make_moons(n_samples=n_samples, noise=0.15, random_state=seed),
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        },
    ),
    (
        datasets.make_blobs(n_samples=n_samples, cluster_std=[1.5, 2.2, 0.5], random_state=random_state),
        {
            "eps": 0.18,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.01,
            "min_cluster_size": 0.2,
        },
    ),
    (
        (np.dot(X, [[0.6, -0.1], [-0.6, 0.8]]), y),
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        },
    ),
    (
        datasets.make_blobs(n_samples=n_samples, random_state=seed),
        {
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2
        }
    ),
    (
        datasets.make_blobs(n_samples=n_samples, centers=6, random_state=random_state),
        {
            "n_clusters": 6,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        }
    ),
    (
        datasets.make_blobs(n_samples=n_samples, centers=4, random_state=random_state),
        {
            "n_clusters": 4,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        }
    ),
    (
        datasets.make_circles(n_samples=n_samples, factor=0.7, noise=0.1, random_state=seed),
        {
            "damping": 0.77,
            "preference": -240,
            "quantile": 0.2,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.08,
        }
    ),
    (
        datasets.make_moons(n_samples=n_samples, noise=0.2, random_state=seed),
        {
            "damping": 0.75,
            "preference": -220,
            "n_clusters": 2,
            "min_samples": 7,
            "xi": 0.1,
        }
    ),
    (
        (np.dot(X, [[0.5, -0.7], [-0.2, 0.9]]), y),
        {
            "eps": 0.15,
            "n_neighbors": 2,
            "min_samples": 7,
            "xi": 0.1,
            "min_cluster_size": 0.2,
        }
    )
]

for i in range(10):
    points, labels = datasets[i][0]
    file = open(rf'datasets/sk-generated/set-{i + 1}/data.csv', mode='w')

    for j in range(n_samples):
        point = points[j]
        label = labels[j]
        file.write(f'{label},{point[0]},{point[1]}\n')

    file.close()
