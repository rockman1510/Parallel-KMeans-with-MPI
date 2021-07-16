import math
import sys
import csv
import time
import numpy as np
import collections
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import adjusted_rand_score

class K_Means:
  def __init__(self, k = 2, tol = 0.001, max_iter = 300):
    self.k = k
    self.tol = tol
    self.max_iter = max_iter
  
  def fit(self, data):
    self.centroids = {}
    # sampl = np.random.randint(0, len(data) - 1,self.k)
    # print(sampl)

    for i in range(3):
      self.centroids[i] = data[i]
    print("centroids: {}".format(self.centroids))
    
    for i in range(self.max_iter):
      self.classifications = {}
      
      for i in range(self.k):
        self.classifications[i] = []
      count = 0
      for featureset in data:
        count = count + 1
        distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
        # if(count < 10):
          # print("featureset: {}".format(featureset))
          # print("distance: {}".format(distances))
          # print("distance min index: {}".format(distances.index(min(distances))))
        classification = distances.index(min(distances))
        self.classifications[classification].append(featureset)
      
      prev_centroids = dict(self.centroids)

      for classification in self.classifications:
        self.centroids[classification] = np.average(self.classifications[classification], axis = 0)
      
      optimized = True
      
      for c in self.centroids:
        original_centroid = prev_centroids[c]
        current_centroid = self.centroids[c]
        # print("current: {}".format(current_centroid))
        # print("previous: {}".format(original_centroid))
        sum_result = np.sum((current_centroid - original_centroid) / original_centroid * 100.0)
        # print("sum result: {}".format(sum_result))
        if sum_result > self.tol:
          optimized = False
      
      if optimized:
        break
  
  def predict(self, data):
    distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
    classification = distances.index(min(distances))
    return classification


num_clusters = int(sys.argv[2])
max_iteration = 5000

with open('kmeans_dataset_1.csv', 'r') as f:
  reader = csv.reader(f)
  data = list(reader)
data_size = int(sys.argv[1])
data = data[0 : data_size]
data = np.array(data).astype(np.float64)

from datetime import datetime

startTime = time.time()
model = K_Means(k = 3, tol = 0.001, max_iter = max_iteration)
model.fit(data[:, :data_size])
endTime = time.time()
print("Executed Time: {} seconds".format((endTime - startTime)))
