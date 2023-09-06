import os
import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from gen import genarateData


def plot(X, centroids, labels, show=True, iteration=None, file_name=None):
    # Plot the original data and clusters
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
    plt.title('K-means Clustering iteration ' + str(iteration))
    plt.xlabel('X')
    plt.ylabel('Y')
    if show:
        plt.show()
    if iteration and file_name:
        file_name = file_name + "_" + str(iteration)
        plt.savefig(file_name)
    plt.close()


class BaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass

class KMeans(BaseModel):
    def __init__(self, n_clusters, max_iter, file_prefix=None) -> None:
        # Initialize KMeans parameters
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._centroids = None
        self._labels = None
        self._file_prefix = file_prefix
        self._init_centroids = None
        self.spend_time = 0
        
        # Create the kmeans_plots folder if it doesn't exist
        input_path = "kmeans_plots"
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        
    @property
    def labels(self):
        return self._labels
    
    @property
    def centroids(self):
        return self._centroids
    
    @property
    def initial_centroids(self):
        return self._init_centroids
    
    def _initialize_centroids(self, K, X):
        # Randomly select initial centroids
        centroid_indices = np.random.choice(len(X), K, replace=False)
        centroids = X[centroid_indices.tolist()]
        self._init_centroids = centroids
        return centroids
    
    def _calculate_euclidean_distance(self, centroids, X):
        # Calculate Euclidean distances between data points and centroids
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return distances
    
    def _assign_labels(self, distances):
        # Assign each data point to the nearest centroid
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X, n_clusters, labels):
        # Update centroids by calculating the mean of data points in each cluster
        new_centroids = []
        for i in range(n_clusters):
            new_centroids.append(np.mean(X[labels==i], axis=0))
        return np.array(new_centroids)    
        
    def fit(self, data, DatasetSize ,plot_graph = False ,y=None):

        start_time = time.time()
        # Load data from CSV file
        try:
            X = np.loadtxt(data, delimiter=',')
        except:
            genarateData(DatasetSize)
            X = np.loadtxt(data, delimiter=',')

        end_time = time.time()

        elapsed_time = end_time - start_time
        print("Reading data from CSV file took %f seconds" % elapsed_time)

        print("Dataset size: ", len(X))

        centroids = self._initialize_centroids(self._n_clusters, X)
        labels = None
        for i in range(self._max_iter):
            startR_time = time.time()
            distances = self._calculate_euclidean_distance(centroids, X)

            labels = self._assign_labels(distances)
        
            centroids = self._update_centroids(X, self._n_clusters, labels)
            endR_time = time.time()
            self.spend_time += endR_time - startR_time

            if plot_graph and self._file_prefix:
                    plot(X, centroids, labels, False, i, self._file_prefix)
        
        print("Calculating KMeans took %f seconds" % self.spend_time)

        self._centroids = centroids
        self._labels = labels
    
    def predict(self, X):
        return NotImplemented("Not implemented")




