import numpy as np
from kmeans import KMeans, plot, comparison_plot  # Import required classes and functions

# Set random seed for reproducibility
np.random.seed(123)

# Generate random data
N = 10000
M = 2
X = np.random.rand(N, M)

# Define parameters for KMeans
K = 3
max_iter = 30

# Create a KMeans instance and fit it to the data
kmeans = KMeans(n_clusters=K, max_iter=max_iter, file_prefix="kmeans_plots/kmeans_clustering")
kmeans.fit(X)

# Generate a GIF animation of the clustering process
from util import generate_gif

# Define input and output paths and file prefix
input_path = "kmeans_plots"
file_prefix = "kmeans_clustering"
output_path = "images"
output_file = "kmeans_clustering_animate.gif"
duration = 1000  # Duration between frames in milliseconds

# Generate the GIF animation
generate_gif(path=input_path, file_prefix=file_prefix, output_path=output_path, output_file=output_file, duration=duration)
