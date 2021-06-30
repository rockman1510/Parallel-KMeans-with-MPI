import math
import csv
import time
import numpy as np
import collections
import threading
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score


# class K_Means:
#   def __init__(self, k = 2, tol = 0.001, max_iter = 300):
#     self.k = k
#     self.tol = tol
#     self.max_iter = max_iter
  
#   def fit(self, data):
#     self.centroids = {}
#     sampl = np.random.randint(0, len(data) - 1,self.k)
#     print(sampl)

#     for i, item in enumerate(sampl):
#       self.centroids[i] = data[item]
#     print("centroids: {}".format(self.centroids))
    
#     for i in range(self.max_iter):
#       self.classifications = {}
      
#       for i in range(self.k):
#         self.classifications[i] = []
#       count = 0
#       for featureset in data:
#         count = count + 1
#         distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
#         classification = distances.index(min(distances))
#         self.classifications[classification].append(featureset)
      
#       prev_centroids = dict(self.centroids)

#       for classification in self.classifications:
#         self.centroids[classification] = np.average(self.classifications[classification], axis = 0)
      
#       optimized = True
      
#       for c in self.centroids:
#         original_centroid = prev_centroids[c]
#         current_centroid = self.centroids[c]
#         if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
#           # print(np.sum((current_centroid - original_centroid) / original_centroid * 100.0))
#           optimized = False
      
#       if optimized:
#         break
  
#   def predict(self, data):
#     distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
#     classification = distances.index(min(distances))
#     return classification


def counter(numbers, thread, result = 0):
	total = 0
	for num in numbers:
		total += num
	print("Thread: {}\tTotal: {}".format(thread, total))
	result.append(total)

def chunkIt(data, thread_number):
	average = len(data) / thread_number
	out = []
	last = 0

	while last < len(data):
		out.append(data[int(last) : int(last + average)])
		last += average
	return out

def calDistance(data, centroids, result):
	clusters = []
	cluster = []
	dist = np.zeros((len(data), len(centroids)))
	for j in range(len(centroids)):
		for k in range(len(data)):
			dist[k][j] = np.linalg.norm(centroids[j] - data[k])

	for h in range (len(dist)):
		clusters.append(np.argmin(dist[h]) + 1)
	result.append(collections.Counter(clusters))	# Counter the elements belong to cluster


def calKmeans(data, centroids, classifications, averageClassify, thread_id):
	# print("Thread_Id: {}\tdata: {}\n".format(thread_id, data))
	localCentroids = {}
	for i in range(len(centroids)):
		classifications[i] = []
	for feature in data:
		distances = [np.linalg.norm(feature - centroids[centroid]) for centroid in centroids]
		classification = distances.index(min(distances))
		classifications[classification].append(feature)
	for classify in classifications:
		if len(classifications[classify]) == 0:
			localCentroids[classify] = np.zeros(3)
		else:
			localCentroids[classify] = np.average(classifications[classify], axis = 0)
	# print("localCentroids: {}\n".format(localCentroids))
	averageClassify.append(localCentroids)


num_threads = 8

print("Input the number of clusters: ")
num_clusters = input()
num_clusters = int(num_clusters)
start_time = time.time()

with open('kmeans_dataset_1.csv', 'r') as f:
	reader = csv.reader(f)
	data = list(reader)

data = data[0 : 10000]
data = np.array(data).astype(np.float64)

kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(data).labels_

initialCentroids = {}
# kRandom = np.random.randint(0, len(data) - 1, num_clusters)
# print("Initial Centroids position in data set: {}".format(kRandom))
# for i, item in enumerate(kRandom):
# 	initialCentroids[i] = data[item]

for i in range(num_clusters):
	initialCentroids[i] = data[i]

print("Initial Centroids: {}\n".format(initialCentroids))
num_points = len(data)
dimensions = len(data[0])
chunks = chunkIt(data, num_threads)
flag = True
previous_result = []
count = 0
totalAverage = []

counter = 0
while flag == True:
	threads = []
	classifications = {}
	averageClassify = []
	for i, chunk in enumerate(chunks):
		threads.append(threading.Thread(target = calKmeans, args = (chunk, initialCentroids, classifications, averageClassify, i)))
	for thread in threads:
		thread.start()
	for thread in threads:
		thread.join()

	previousCentroids = initialCentroids
	globalClassifications = {}
	for i in range(len(initialCentroids)):
		globalClassifications[i] = []

	newCentroids = {}
	# print("averageClassify: {}".format(averageClassify))
	for i in range(len(averageClassify)):
		for j in averageClassify[i]:
			globalClassifications[j].append(averageClassify[i][j]) 
	for globalClassify in globalClassifications:
		newCentroids[globalClassify] = np.average(globalClassifications[globalClassify], axis = 0)

	# print("previousCentroids: {}".format(previousCentroids))
	# print("newCentroids: {}".format(newCentroids))
	

	counter += 1
	flag = False
	for i in range(len(initialCentroids)):
		originalCentroid = previousCentroids[i]
		currentCentroid = newCentroids[i]
		if np.sum((currentCentroid - originalCentroid) / originalCentroid * 100.0) > 0.001:
			print(np.sum((currentCentroid - originalCentroid) / originalCentroid * 100.0))
			flag = True
	initialCentroids = newCentroids
	# print("initialCentroids: {}".format(initialCentroids))
end_time = time.time()
print("Time executed: {}".format(end_time - start_time))
print("counter: {}".format(counter))
