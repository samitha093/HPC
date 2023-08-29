import numpy as np

def genarateData():
    # Set random seed for reproducibility
    np.random.seed(123)

    # Generate random data
    N = 10000 #data points
    M = 2 #features
    X = np.random.rand(N, M)

    # Save data to CSV file
    np.savetxt('data.csv', X, delimiter=',')

genarateData()
