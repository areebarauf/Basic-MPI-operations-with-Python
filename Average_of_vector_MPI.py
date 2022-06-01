"""
Find an average of numbers in a vector.
"""

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
number_of_processes = comm.Get_size()
rank = comm.Get_rank()
root = 0
start_time=MPI.Wtime()
if rank == root:
    vector = np.random.randint(10, size=10)
    # vector = np.arange(1, 10)
    total_sum = 0
    #print("vector:", vector)

    # Create chunks for each process
    chunks = np.array_split(vector, number_of_processes)
    #print("chunks:", chunks)

    # Send each process their chunk
    for process_rank in range(number_of_processes):
        if process_rank == rank:
            continue
        else:
            comm.send(chunks[process_rank], dest=process_rank)

    # Compute one chunk on the root process
    #print(f"Process {root} should work on chunk: {chunks[root]}")
    computed_sum = chunks[root].sum()
    #print(f"Process {root} computed sum: {computed_sum}\n")
    total_sum += computed_sum

    # Receive computed result from each process
    for process_rank in range(number_of_processes):
        if process_rank == rank:
            continue
        else:
            received_computed_result = comm.recv(source=process_rank)
            print(f"Process {rank} received computed sum from {process_rank}: {received_computed_result}")
            total_sum += received_computed_result
            print(f"\nTotal sum is {total_sum}")

    # Perform the final division on the root process to calculate the average
    average = total_sum / vector.size
    print("Average equals:", average)

    # Test: Compute average sequentially to verify that our parallel average is correct
    sequential_average = vector.sum() / vector.size
    print("Sequential average equals:", sequential_average)

else:
    received_chunk = comm.recv(source=root)
    #print(f"Process {rank} received chunk: {received_chunk}")

    # Compute chunk on the current process
    computed_sum = np.array(received_chunk).sum()
    #print(f"Process {rank} computed sum: {computed_sum}\n")

    # Send computed sum back to root
    comm.send(computed_sum, dest=root)
end_time=MPI.Wtime()

Net_time=end_time-start_time
print('Total Time for process: rank=%.1f is Net_time=%.3f' % (rank, Net_time))