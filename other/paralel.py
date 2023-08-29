from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time
import warnings

# Initialize MPI
comm = MPI.COMM_WORLD  # Initialize MPI communicator
rank = comm.Get_rank()  # Get the rank (identifier) of the current process
size = comm.Get_size()  # Get the total number of processes in the communicator

# Data to scatter init
data_size = 1000000 // size
index_start = rank * data_size
if index_start != 0:
    index_start += 1
index_end = (rank + 1) * data_size
    
startR_time = time.time()
# Read rows of data.csv
data = np.genfromtxt('generated_data.csv', delimiter=',', skip_header=index_start, max_rows=data_size)
endR_time = time.time()

# calculate time to read data
elapsedR_time = endR_time - startR_time
print(f"Process {rank}: Reading data took {elapsedR_time:.4f} seconds ")

# Find the maximum reading time among all processes
max_elapsedR_time = comm.allreduce(elapsedR_time, op=MPI.MAX)

if rank == 0:
    # Print maximum reading time on all processes
    print(f"Process {rank}: Maximum reading time among all processes: {max_elapsedR_time:.4f} seconds")

# Start measuring time
start_time = time.time()

# Apply K-means clustering with n_init set explicitly
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=FutureWarning)
    kmeans = KMeans(n_clusters=3, n_init=10)
    kmeans.fit(data)
labels_local = kmeans.labels_
centroids_local = kmeans.cluster_centers_

# End measuring time
end_time = time.time()
elapsed_time = end_time - start_time

# Print elapsed time for each process
print(f"Process {rank}: K-means clustering took {elapsed_time:.4f} seconds ")

# Print local K-means labels and centroids
# print(f"Process {rank}: Local K-means data set size: {len(data)}")
# print(f"Process {rank}: Local K-means centroids:\n{centroids_local}")

# # Save local K-means figure for each process
# plt.scatter(data[:, 0], data[:, 1], c=labels_local, s=50, cmap='viridis')
# plt.scatter(centroids_local[:, 0], centroids_local[:, 1], c='red', marker='X', s=200)
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.title(f'Local K-means Clustering - Process {rank}')
# local_figure_filename = f'local_kmeans_figure_{rank}.png'
# plt.savefig(local_figure_filename)
# plt.close()
# print(f"Process {rank}: Local K-means figure saved as {local_figure_filename}")

# collect k means arrays
all_centroids_array = comm.gather(centroids_local, root=0)

# Calculate average centroids on root process
if rank == 0:
    all_centroids = np.concatenate(all_centroids_array)
    # Apply K-means clustering with n_init set explicitly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        kmeans = KMeans(n_clusters=3, n_init=10)
        kmeans.fit(all_centroids)
    labels_final = kmeans.labels_
    centroids_final = kmeans.cluster_centers_
    # Print local K-means labels and centroids
    print(f"Process {rank}: final K-means data set size: {len(all_centroids)}")
    print(f"Process {rank}: final K-means centroids:\n{centroids_final}")

    all_end_time = time.time()
    all_elapsed_time = all_end_time - start_time
    print(f"Process {rank}: Final K-means clustering took {all_elapsed_time:.4f} seconds")

    # # Save local K-means final figure
    # plt.scatter(data[:, 0], data[:, 1], c=labels_final, s=50, cmap='viridis')
    # plt.scatter(centroids_final[:, 0], centroids_final[:, 1], c='red', marker='X', s=200)
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.title(f'Final K-means')
    # final_figure_filename = f'Final_kmeans_figure.png'
    # plt.savefig(final_figure_filename)
    # plt.close()
    # print(f"Process {rank}: Final K-means figure saved as {final_figure_filename}")


# Finalize MPI
MPI.Finalize()
