import math
import sys
import csv
import time
import numpy as np
import collections
from mpi4py import MPI
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

def chunkIt(data, num_processors):
	length_per_process = len(data) / float(num_processors)
	out = []
	first = 0
	last = first + int(length_per_process)
	print("first:{} - last:{}".format(first, last))
	for i in range(num_processors):
		out.append(data[first : last])
		first = last
		last += int(length_per_process)
		if len(data) - last < int(length_per_process):
			last = len(data)

	# while last < len(data):
	# 	out.append(data[int(last) : int(last + length_per_process)])
	# 	last += length_per_process
	return out

def addCounter(counter1, counter2, datatype):
    for item in counter2:
        if item in counter1:
            counter1[item] += counter2[item]
        else:
            counter1[item] = counter2[item]
    return counter1


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

global dimensions, num_clusters, num_points, data, flag
num_clusters = 0

if rank == 0:
	if len(sys.argv) != 3:
		print("Please input follow the format: mpiexec -n [number of processors] python [python file] [number of data] [number of clusters]")

	if not isinstance(int(sys.argv[1]), int) :
		print("Invalid number of data! - (Integer)")
		print("Please input follow the format: mpiexec -n [number of processors] python [python file] [number of data] [number of clusters]")

	if not isinstance(int(sys.argv[2]), int):
		print("Invalid number of clusters! - (Integer)")
		print("Please input follow the format: mpiexec -n [number of processors] python [python file] [number of data] [number of clusters]")

	num_clusters = int(sys.argv[2])
	start_time = time.time()

	with open('kmeans_dataset_1.csv', 'r') as f:
		reader = csv.reader(f)
		data = list(reader)
	data_size = int(sys.argv[1])
	data = data[0 : data_size]
	data = np.array(data).astype(np.float64)

	kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(data).labels_

	initialCentroid = []
	# kRandom = np.random.randint(0, len(data) - 1, num_clusters)
	# print("initialCentroid Centroids position in data set: {}".format(kRandom))
	# for i in kRandom:
	# 	initialCentroid.append(data[i])

	for i in range(num_clusters):
		initialCentroid.append(data[i])
		
	initialCentroid = np.vstack(initialCentroid)
	chunks = chunkIt(data, size)
else:
	chunks = None
	initialCentroid = None
	data = None
	num_clusters = None
	centroid = None
	start_time = None
start_time = comm.bcast(start_time, root = 0)
data = comm.scatter(chunks, root = 0)
num_clusters = comm.bcast(num_clusters, root = 0)
if rank == 0:
	print("initialCentroid cluster: {}".format(initialCentroid))
initialCentroid = comm.bcast(initialCentroid, root = 0)
flag = True

while flag == True:
	clusters = []
	cluster = []
	distance = np.zeros((len(data), len(initialCentroid)))
	for j in range(len(initialCentroid)):
		for k in range(len(data)):
			distance[k][j] = np.linalg.norm(initialCentroid[j] - data[k])

	for h in range (len(distance)):
		clusters.append(np.argmin(distance[h]) + 1)
	clusterCounter = collections.Counter(clusters)	# Counter the elements belong to cluster
	counterSumOperation = MPI.Op.Create(addCounter, commute = True)	# Create operation with a method
	totalCounter = comm.allreduce(clusterCounter, op = counterSumOperation)	# Apply operation in allreduce
	comm.Barrier()	# make a synchronous point for all processors 
	cluster = comm.gather(clusters, root = 0)
	comm.Barrier()
	if rank == 0:
		cluster = [item for sublist in cluster for item in sublist]
	centroids = np.zeros((len(initialCentroid), len(initialCentroid[0])))
	for z in range (1, num_clusters + 1):
		indices = [a for a, b in enumerate(clusters) if b == z]
		centroids[z - 1] = np.divide((np.sum([data[a] for a in indices], axis = 0)).astype(np.float64), totalCounter[z])
	centroid = comm.allreduce(centroids, MPI.SUM)
	comm.Barrier()

	if np.all(centroid == initialCentroid):
		flag = False
	else:
		initialCentroid = centroid
	comm.Barrier()
if rank == 0:
	print("\n===================================Result===================================")
	print("final cluster: {}".format(initialCentroid))
	print("Execution time %s seconds" % (time.time() - start_time))
	print("Adjusted Rank Score", adjusted_rand_score(kmeans, cluster))
MPI.Finalize()		
exit(0)	