from sklearn.datasets import make_blobs

# Generate sample data
n_samples = 1000000
n_clusters = 3
X, y_true = make_blobs(n_samples=n_samples, n_features=2, centers=n_clusters, random_state=42)

# Save data to CSV file
data_filename = 'generated_data.csv'
np.savetxt(data_filename, X, delimiter=',')
