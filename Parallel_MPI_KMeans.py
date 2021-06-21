import math
import csv
import time
import numpy as np
import collections
from mpi4py import MPI
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

def chunkIt(data, num_processors):
	average = len(data) / float(num_processors)
	out = []
	last = 0.0

	while last < len(data):
		out.append(data[int(last) : int(last + average)])
		last += average
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
	print("Input the number of clusters: ")
	num_clusters = input()
	num_clusters = int(num_clusters)
	start_time = time.time()

	with open('kmeans_dataset.csv', 'r') as f:
		reader = csv.reader(f)
		data = list(reader)

	data.pop(0)
	for i in range (len(data)):
		data[i].pop(0)
	data = data[0 : 10000]
	data = np.array(data).astype(np.float64)

	kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(data).labels_

	initial = []
	kRandom = np.random.randint(0, len(data) - 1, num_clusters)
	print("Initial Centroids position in data set: {}".format(kRandom))
	for i in kRandom:
		initial.append(data[i])
	initial = np.vstack(initial)
	num_points = len(data)
	dimensions = len(data[0])
	chunks = chunkIt(data, size)
else:
	chunks = None
	initial = None
	data = None
	dimensions = None
	num_points = None
	num_clusters = None
	centroid = None
	start_time = None
start_time = comm.bcast(start_time, root = 0)
data = comm.scatter(chunks, root = 0)
num_clusters = comm.bcast(num_clusters, root = 0)
if rank == 0:
	print("initial cluster: {}".format(initial))
initial = comm.bcast(initial, root = 0)
flag = True

while flag == True:
	clusters = []
	cluster = []
	dist = np.zeros((len(data), len(initial)))
	for j in range(len(initial)):
		for k in range(len(data)):
			dist[k][j] = np.linalg.norm(initial[j] - data[k])

	for h in range (len(dist)):
		clusters.append(np.argmin(dist[h]) + 1)
	ClustCounter = collections.Counter(clusters)	# Counter the elements belong to cluster
	counterSumOp = MPI.Op.Create(addCounter, commute = True)	# Create operation with a method
	totalcounter = comm.allreduce(ClustCounter, op = counterSumOp)	# Apply operation in allreduce
	comm.Barrier()	# make a synchronous point for all processors 
	cluster = comm.gather(clusters, root = 0)
	comm.Barrier()
	if rank == 0:
		cluster = [item for sublist in cluster for item in sublist]
	centroids = np.zeros((len(initial), len(initial[0])))
	for z in range (1, num_clusters + 1):
		indices = [a for a, b in enumerate(clusters) if b == z]
		centroids[z - 1] = np.divide((np.sum([data[a] for a in indices], axis = 0)).astype(np.float64), totalcounter[z])
	centroid = comm.allreduce(centroids, MPI.SUM)
	comm.Barrier()

	if np.all(centroid == initial):
		flag = False
	else:
		initial = centroid
	comm.Barrier()
if rank == 0:
	print("\n===================================Result===================================")
	print("final cluster: {}".format(initial))
	print("Execution time %s seconds" % (time.time() - start_time))
	print("Adjusted Rank Score", adjusted_rand_score(kmeans, cluster))
MPI.Finalize()		
exit(0)	