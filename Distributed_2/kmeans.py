import os
import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC
from abc import abstractmethod
from mpi4py import MPI
from get import RabbitMQConsumer

def plot(X, centroids, labels, show=True, iteration=None, file_name=None):
    """
    The function `plot` is used to plot the original data and clusters in a K-means clustering
    algorithm, with the option to save the plot to a file.
    
    Args:
      X: The X parameter is a numpy array that represents the data points to be plotted. Each row of the
    array represents a data point, and the columns represent the features of the data point.
      centroids: The centroids parameter is a numpy array that represents the coordinates of the
    centroids in the clustering algorithm. Each row of the array represents the coordinates of a
    centroid.
      labels: The "labels" parameter is a list or array that assigns each data point in X to a specific
    cluster. Each element in the "labels" list corresponds to a data point in X and indicates which
    cluster that data point belongs to.
      show: The "show" parameter is a boolean value that determines whether the plot should be displayed
    or not. If set to True, the plot will be displayed using the plt.show() function. If set to False,
    the plot will not be displayed. Defaults to True
      iteration: The iteration parameter is used to indicate the current iteration number of the K-means
    clustering algorithm. It is used to update the title of the plot to show the iteration number.
      file_name: The `file_name` parameter is a string that specifies the name of the file to save the
    plot as. If provided, the plot will be saved as an image file with the specified name. If not
    provided, the plot will not be saved as a file.
    """
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

# Abstract base class for a machine learning model that defines the fit and
# predict methods.
class BaseModel(ABC):
    
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def fit(self, X, y):
        pass
    
    @abstractmethod
    def predict(self, X):
        pass
    
    
# The `KMeans` class implements the K-means clustering algorithm using RabbitMQ for data input.
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
        self.spend_com_time = 0
        self.spend_read_time = 0
        self.consumer = None

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
        The function initializes the centroids for the K-means clustering algorithm by randomly selecting K
        data points from the input array X.
        
        Args:
          K (int): The number of centroids to initialize.
          X (np.array): X is a numpy array containing the data points.
        
        Returns:
          the centroids, which is a numpy array.
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
        The function calculates the Euclidean distance between each point in X and each centroid in
        centroids.
        
        Args:
          centroids (np.array): A numpy array representing the centroids of the clusters. Each row of the
        array represents the coordinates of a centroid.
          X (np.array): X is a numpy array representing the data points. Each row of X represents a data
        point, and each column represents a feature of that data point.
        
        Returns:
          an array of distances between each point in X and each centroid in centroids.
        """
        distances = np.linalg.norm(X[:, None] - centroids, axis=2)
        return distances
    
    def _assign_labels(self, distances: np.array) -> np.array:
        """
        The function `_assign_labels` takes in a numpy array of distances and returns an array of labels
        corresponding to the minimum distance for each row.
        
        Args:
          distances (np.array): The `distances` parameter is a numpy array that represents the distances
        between data points and cluster centroids. Each row of the array corresponds to a data point, and
        each column corresponds to a cluster centroid. The values in the array represent the distances
        between the data points and the cluster centroids.
        
        Returns:
          an array containing the indices of the minimum values along each row of the input array
        "distances".
        """
        return np.argmin(distances, axis=1)
    
    def _update_centroids(self, X: np.array, n_clusters: int, labels: np.array) -> np.array:
        """
        The function `_update_centroids` calculates the new centroids for each cluster based on the mean of
        the data points belonging to that cluster.
        
        Args:
          X (np.array): X is a numpy array representing the data points. Each row of X represents a data
        point, and each column represents a feature of that data point. The shape of X is (number of data
        points, number of features).
          n_clusters (int): The parameter `n_clusters` represents the number of clusters in the dataset. It
        is an integer value that specifies the desired number of clusters to be formed.
          labels (np.array): The `labels` parameter is a numpy array that contains the cluster assignments
        for each data point in the input `X`. Each element in the `labels` array represents the cluster
        index to which the corresponding data point in `X` belongs.
        
        Returns:
          The function `_update_centroids` returns a numpy array `new_centroids` which contains the updated
        centroids for each cluster.
        """
        new_centroids = []
        for i in range(n_clusters):
            # local mean of the centroid belonging to cluster `i`
            if len(X[labels==i]) == 0:
                print(f"Process {self._rank}: No data points in cluster {i}")
                local_centroid = np.zeros(X.shape[1])
            else:
                local_centroid = np.mean(X[labels==i], axis=0) 
            start_com_time = time.time()
            # global sum of local mean of the centroid belonging to cluster `i`
            new_centroid_imd = self._comm.allreduce(local_centroid, op=MPI.SUM)
            end_com_time = time.time()
            elapsed_com_time = end_com_time - start_com_time
            self.spend_com_time += elapsed_com_time
            # mean value of the global sum taken by dividing with number of total processes
            new_centroid = new_centroid_imd / self._size
            new_centroids.append(new_centroid) 
        return np.array(new_centroids)    
    
    def update_data(self,itteration):
        """
        The function `update_data` connects to a consumer, starts consuming messages, retrieves the received
        messages, converts them to a numpy array, and returns the array.
        
        Args:
          itteration: The `iteration` parameter is used to specify the iteration number for which the data
        is being updated. It is likely used within the `start_consuming` method to determine which messages
        to consume.
        
        Returns:
          the variable `x_new`, which is an array of floating-point numbers.
        """
        # Connect and start consuming messages
        self.consumer.memoryDown()
        self.consumer.connect()
        self.consumer.start_consuming(self._rank,itteration)
        # Retrieve the received messages
        recived_data = self.consumer.get_received_messages()
        x_new = np.array(recived_data, dtype=float)
        return x_new
        
    def fit(self, DataQueue, username, password , host , batchSetSize ,plot_graph = False ,y=None) -> None:
        """
        The `fit` function performs the k-means clustering algorithm on a given dataset, using RabbitMQ
        for data communication between processes, and outputs the centroids and labels.
        
        Args:
          DataQueue: DataQueue is the name of the RabbitMQ queue from which the data will be consumed.
          username: The username is the username used to authenticate with the RabbitMQ server.
          password: The `password` parameter is used to provide the password for authentication when
        connecting to the RabbitMQ server.
          host: The `host` parameter is the hostname or IP address of the RabbitMQ server that you want
        to connect to.
          batchSetSize: The parameter `batchSetSize` represents the size of each batch of data that will
        be processed at a time. It determines how many data records will be loaded and processed in each
        iteration of the algorithm.
          plot_graph: The `plot_graph` parameter is a boolean flag that determines whether to create
        plots at each iteration of the calculation. If set to `True`, plots will be created. If set to
        `False`, plots will not be created. Defaults to False
          y: The parameter `y` is not used in the `fit` method. It is not necessary for the execution of
        the method.
        """
        # # load data
        start_read_time = time.time()
        # Create a RabbitMQConsumer instance with the provided credentials and host
        self.consumer = RabbitMQConsumer(DataQueue, username, password, host, batchSetSize)

        # Connect and start consuming messages
        self.consumer.connect()
        self.consumer.start_consuming(self._rank, 0)

        # Retrieve the received messages
        recived_data = self.consumer.get_received_messages()
        x_local = np.array(recived_data, dtype=float)

        end_read_time = time.time()
        elapsed_read_time = end_read_time - start_read_time
        self.spend_read_time = elapsed_read_time
        print(f"Process {self._rank}: Data loader took {elapsed_read_time:.4f} seconds")
        
        # initialize centroids
        centroids = self._initialize_centroids(self._n_clusters, x_local)
        #wait for all processes to initialize centroids
        print(f"Process {self._rank}: Waiting for all processes to initialize centroids")
        
        start_time = time.time()
        print(f"Process {self._rank}: Starting calculation")
        labels = None
        for i in range(self._max_iter):
            distances = self._calculate_euclidean_distance(centroids, x_local)
            print(f"Process {self._rank}: distances calculated in itteration {i}")

            labels = self._assign_labels(distances)
            print(f"Process {self._rank}: labels assigned in itteration {i}")
        
            centroids = self._update_centroids(x_local, self._n_clusters, labels)
            print(f"Process {self._rank}: centroids updated in itteration {i}")

            # If file_prefix is provided, create plots at each iteration
            if plot_graph and self._rank == 0 and self._file_prefix:
                plot(x_local, centroids, labels, False, i, self._file_prefix)
            
            new_data = self.update_data(i+1)
            if new_data.size != 0:
                x_local = np.concatenate((x_local, new_data), axis=0)
                print(f"Process {self._rank}: apeend new {len(new_data)} data records. Total data array size: {len(x_local)} in itteration {i}")
                
        end_time = time.time()
        print(f"Process {self._rank}: Calculation completed")
        elapsed_time = end_time - start_time
        self.spend_time = elapsed_time

        print(f"Process {self._rank}: Calculation took {self.spend_time - self.spend_com_time:.4f} seconds")
        print(f"Process {self._rank}: Communication took {self.spend_com_time:.4f} seconds")

        self._centroids = centroids
        self._labels = labels

        final_spend_read_time = self._comm.allreduce(self.spend_read_time, op=MPI.MIN)
        final_spend_com_time = self._comm.allreduce(self.spend_com_time, op=MPI.MAX)
        final_spend_calc_time = self._comm.allreduce(self.spend_time - self.spend_com_time, op=MPI.MAX)

        if self._rank == 0:
            print(f"\033[1;32mData loader took {final_spend_read_time:.4f} seconds\033[0m")
            print(f"\033[1;33mCommunication took {final_spend_com_time:.4f} seconds\033[0m")
            print(f"\033[1;31mCalculation took {final_spend_calc_time:.4f} seconds\033[0m")

    
    def predict(self, X: np.array) -> np.array:
        """
        The function predict takes in an array X and returns a placeholder value indicating that it is not
        implemented yet.
        
        Args:
          X (np.array): An input array of shape (n_samples, n_features) where n_samples is the number of
        samples and n_features is the number of features.
        
        Returns:
          The `NotImplemented` object is being returned.
        """
        return NotImplemented("Not implemented")



