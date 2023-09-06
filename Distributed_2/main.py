import numpy as np
from kmeans import KMeans
from mpi4py import MPI

# Load data from CSV file
username = 'admin'
password = 'admin'
host = '172.28.128.1'
DataQueue = "K_queue"
batchSetSize = 10

# Define parameters for KMeans
K = 3 #number of clusters
max_iter = 30 #maximum number of iterations

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Create a KMeans instance and fit it to the data
kmeans = KMeans(n_clusters=K, max_iter=max_iter, comm=comm, file_prefix="kmeans_plots/kmeans_clustering")
kmeans.fit(DataQueue, username, password, host, batchSetSize, True)

# Generate a GIF animation of the clustering process
from util import generate_gif

if rank == 0:
    # Define input and output paths and file prefix
    input_path = "kmeans_plots"
    file_prefix = "kmeans_clustering"
    output_path = "images"
    output_file = "kmeans_clustering_animate.gif"
    duration = 1000 # duration of each frame in milliseconds

    # Generate the GIF animation
    try:
        generate_gif(path=input_path, file_prefix=file_prefix, output_path=output_path, output_file=output_file, duration=duration) 
    except:
        print("Error: Could not generate GIF animation")

# mpirun -n 4 python3 main.py