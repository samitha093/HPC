from kmeans import KMeans

# Load data from CSV file
DataFile = "data.csv"
DataSetSize = 5*1000000 #5 million data points

# Define parameters for KMeans
K = 3 #number of clusters
max_iter = 30 #maximum number of iterations

# Create a KMeans instance and fit it to the data
kmeans = KMeans(n_clusters=K, max_iter=max_iter, file_prefix="kmeans_plots/kmeans_clustering")
kmeans.fit(DataFile, DataSetSize, False)

# Generate a GIF animation of the clustering process
from util import generate_gif

# Define input and output paths and file prefix
input_path = "kmeans_plots"
file_prefix = "kmeans_clustering"
output_path = "images"
output_file = "kmeans_clustering_animate.gif"
duration = 1000  # Duration between frames in milliseconds

# Generate the GIF animation
try:
    generate_gif(path=input_path, file_prefix=file_prefix, output_path=output_path, output_file=output_file, duration=duration) 
except:
    print("Error: Could not generate GIF animation")    