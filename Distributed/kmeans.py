import os
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from abc import abstractmethod
from mpi4py import MPI

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
    
    def __init__(self, n_clusters, max_iter, comm, file_prefix=None) -> None:
        self._n_clusters = n_clusters
        self._max_iter = max_iter
        self._centroids = None
        self._labels = None
        self._file_prefix = file_prefix
        self._initial_centroids = None
        self._comm = comm
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()

        # Create the kmeans_plots folder if it doesn't exist
        input_path = "kmeans_plots"
        if not os.path.exists(input_path):
            os.makedirs(input_path)
        
    @property
    def lables(self):
        return self._labels
    
    @property
    def centroids(self):
        return self._centroids
    
    @property
    def initial_centroids(self):
        return self._initial_centroids
    
    def _initialize_centroids(self, K:int, X:np.array) -> np.array:
        """
        Calculates the initial centroids

        Args:
            K (int): number of clusters
            X (np.array): training data

        Returns:
            np.array: initial centroids
        """
        centroids = None
        if self._rank == 0:
            centroid_indices = np.random.choice(len(X), K, replace=False)
            centroids = X[centroid_indices.tolist()]
        centroids = self._comm.bcast(centroids, root=0)
        self._initial_centroids = centroids
        return centroids
    
    def _calculate_euclidean_distance(self, centroids: np.array, X: np.array) -> np.array:
        """
        Calculates the euclidean distance of centroids and data
        
        Args:
            centroids (np.array): cluster centroids
            X (np.array): data

        Returns:
            np.array: distance as an array
        """
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return distances
    
    def _assign_labels(self, distances: np.array) -> np.array:
        """
        Assign labels for data points

        Args:
            distances (np.array): euclidean distances

        Returns:
            np.array: labels as an integer array
        """
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.array, n_clusters: int, labels: np.array) -> np.array:
        """
        Update centroids
        Args:
            X (np.array): data
            n_clusters (int): number of clusters
            labels (np.array): cluster id per each data point

        Returns:
            np.array: updated centroids
        """
        new_centroids = []
        for i in range(n_clusters):
            # local mean of the centroid belonging to cluster `i`
            local_centroid = np.mean(X[labels==i], axis=0) 
            # global sum of local mean of the centroid belonging to cluster `i`
            new_centroid_imd = self._comm.allreduce(local_centroid, op=MPI.SUM)
            # mean value of the global sum taken by dividing with number of total processes
            new_centroid = new_centroid_imd / self._size
            new_centroids.append(new_centroid) 
        return np.array(new_centroids)    
        
    def fit(self, X: np.array, y=None) -> None:
        """
        Training the KMeans algorithm

        Args:
            X (np.array): data
            y : Ignored but placed as a convention.
        """
        # initialize centroids
        centroids = self._initialize_centroids(self._n_clusters, X)
        
        # scatter data
        x_local = np.empty((X.shape[0]//self._size, X.shape[1]), dtype=X.dtype)
        self._comm.Scatter(X, x_local, root=0)
        labels = None
        for i in range(self._max_iter):
            distances = self._calculate_euclidean_distance(centroids, x_local)

            labels = self._assign_labels(distances)
        
            centroids = self._update_centroids(x_local, self._n_clusters, labels)

            # If file_prefix is provided, create plots at each iteration
            if self._file_prefix:
                plot(X, centroids, labels, False, i, self._file_prefix)

        self._centroids = centroids
        self._labels = labels
    
    def predict(self, X: np.array) -> np.array:
        return NotImplemented("Not implemented")



