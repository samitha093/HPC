import numpy as np
from sklearn.datasets import make_blobs

# Generate sample data with larger variations
n_samples = 10000000
n_clusters = 3
cluster_std = [3.0, 4.0, 2.5]
X, y_true = make_blobs(n_samples=n_samples, n_features=2, centers=n_clusters, cluster_std=cluster_std, random_state=42)

# Save data to CSV file
data_filename = 'generated_data.csv'
np.savetxt(data_filename, X, delimiter=',')
