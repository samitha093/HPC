import numpy as np

def genarateData(size=10, features=2):
    """
    The function `generateData` generates random data of a specified size and number of features, and
    saves it to a CSV file.
    
    Args:
      size: The size parameter determines the number of data points to generate. In this case, it is set
    to 10, so the function will generate 10 data points. Defaults to 10
      features: The "features" parameter specifies the number of features or columns in the generated
    data. Defaults to 2
    """
    # Set random seed for reproducibility
    np.random.seed(123)

    # Generate random data
    X = np.random.rand(size, features)

    # Save data to CSV file
    np.savetxt('data.csv', X, delimiter=',')
