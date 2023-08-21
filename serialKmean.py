import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import time

# Define the number of clusters
n_clusters = 3

# Load data from CSV file
data = np.loadtxt('generated_data.csv', delimiter=',')

# Start measuring time
start_time = time.time()

# Apply K-means clustering
kmeans = KMeans(n_clusters=n_clusters)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# End measuring time
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"K-means clustering took {elapsed_time:.4f} seconds")

# Visualize the clusters and centroids
plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.show()
