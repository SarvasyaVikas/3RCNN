import numpy as np
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

start_time = time.time()
tot = 0
lower = (10 ** 7) * 2 * rank
upper = (10 ** 7) * 2 * (rank + 1)
for j in range(lower, upper):
	tot += 1
stop_time = time.time()
print("parallel")
print(stop_time - start_time)
