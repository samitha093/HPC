import numpy as np

def genarateData(size=10000, features=2):
    # Set random seed for reproducibility
    np.random.seed(123)

    # Generate random data
    X = np.random.rand(size, features)

    # Save data to CSV file
    np.savetxt('data.csv', X, delimiter=',')

# genarateData()
