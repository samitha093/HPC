from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Generate random data
np.random.seed(123)
local_data = np.random.rand(2, 2)

# Gather data from all processes
gathered_data = None
if rank == 0:
    gathered_data = np.empty((16, 2), dtype=float)  # Correct size and dtype

comm.Gather(local_data, gathered_data, root=0)

# Print the gathered data on the root process
if rank == 0:
    print(f"Rank {rank}: Gathered data:\n{len(gathered_data)}")
