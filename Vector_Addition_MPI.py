from mpi4py import MPI
import numpy as np


def add_vectors(v1, v2):
    _result = []
    for element_of_vector1, element_of_vector2 in zip(v1, v2):
        _result.append(element_of_vector1 + element_of_vector2)
    return _result


comm = MPI.COMM_WORLD
number_of_processes = comm.Get_size()
rank = comm.Get_rank()
root = 0
N=10**6
start_time=MPI.Wtime()

if rank == root:
    # Create vectors
    vector1 = np.random.randint(10, size=N)
    vector2 = np.random.randint(10, size=N)
    result = np.array([])

    # Create chunks for each process
    chunks_of_vector1 = np.array_split(vector1, number_of_processes)
    chunks_of_vector2 = np.array_split(vector2, number_of_processes)

    # Send each process their chunk
    for process_rank in range(number_of_processes):
        if process_rank == rank:
            continue
        else:
            comm.send(chunks_of_vector1[process_rank], dest=process_rank)
            comm.send(chunks_of_vector2[process_rank], dest=process_rank)

    # Compute one chunk on the root process
    computed_result = add_vectors(chunks_of_vector1[root], chunks_of_vector2[root])
    # print(f"Process {root} computed result: {computed_result}\n")
    result = np.append(result, computed_result)

    # Receive computed result from each process
    for process_rank in range(number_of_processes):
        if process_rank == rank:
            continue
        else:
            received_computed_result = comm.recv(source=process_rank)
            result = np.append(result, received_computed_result)
            print('parallel added result:',result)

    # Test: use numpy to compute the result sequentially and verify that both results match
    sequential_result = np.add(vector1, vector2)
    print('sequential added result:',sequential_result)
    #print('Sequential Addition:',sequential_result)
    if np.array_equal(result, sequential_result):
        print("Test passes!")
    else:
        print("Test fails")

else:

    received_chunk1 = comm.recv(source=root)
    received_chunk2 = comm.recv(source=root)
    # print(f"Process {rank} received chunk1: {received_chunk1}")
    # print(f"Process {rank} received chunk2: {received_chunk2}")

    # Compute chunk on the current process
    result = add_vectors(received_chunk1, received_chunk2)
    # print(f"Process {rank} computed result: {result}\n")

    # Send computed result back to root
    comm.send(result, dest=root)

end_time=MPI.Wtime()

Net_time=end_time-start_time
print('Total Time for process: rank=%.1f is Net_time=%.3f' % (rank, Net_time))
# print('Total Time Taken for Process:{}, is {}',number_of_processes,Net_time)