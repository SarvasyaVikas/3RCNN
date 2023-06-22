import time
from mpi4py import MPI
start = time.time()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
middle = time.time()
print(rank)
end = time.time()
print(end - start)
print(end - middle)
print(middle - start)
