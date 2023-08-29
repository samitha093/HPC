import os
import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from abc import abstractmethod
from mpi4py import MPI
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
        self.spend_time = 0

        if self._rank == 0:
            # Create the kmeans_plots folder if it doesn't exist
            input_path = "kmeans_plots"
            if not os.path.exists(input_path):
                os.makedirs(input_path)
            # Create the images folder if it doesn't exist
            output_path = "images"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
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
        
    def fit(self, data, DatasetSize ,plot_graph = False ,y=None) -> None:
        """
        Training the KMeans algorithm

        Args:
            X (String): data file path
            y : Ignored but placed as a convention.
        """
        # Data to scatter init
        data_size = DatasetSize // self._size
        index_start = self._rank * data_size
        if index_start != 0:
            index_start += 1
            
        startR_time = time.time()
        # Read rows from data.csv
        try:
            x_local = np.genfromtxt(data, delimiter=',', skip_header=index_start, max_rows=data_size)
        except:
            genarateData(DatasetSize)
            x_local = np.genfromtxt(data, delimiter=',', skip_header=index_start, max_rows=data_size)
        endR_time = time.time()

        # calculate time to read data
        elapsedR_time = endR_time - startR_time
        print(f"Process {self._rank}: Reading data took {elapsedR_time:.4f} seconds")

        # initialize centroids
        centroids = self._initialize_centroids(self._n_clusters, x_local)
        
        # calculation data loop
        labels = None
        for i in range(self._max_iter):

            start_time = time.time()

            distances = self._calculate_euclidean_distance(centroids, x_local)

            labels = self._assign_labels(distances)
        
            centroids = self._update_centroids(x_local, self._n_clusters, labels)

            end_time = time.time()
            self.spend_time += end_time - start_time

            if plot_graph and self._rank == 0 and self._file_prefix:
                        plot(x_local, centroids, labels, False, i, self._file_prefix)
        
        print(f"Process {self._rank}: Calculation took {self.spend_time:.4f} seconds")

        self._centroids = centroids
        self._labels = labels
    
    def predict(self, X: np.array) -> np.array:
        return NotImplemented("Not implemented")
    


