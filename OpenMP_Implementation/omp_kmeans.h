#ifndef OMP_KMEANS
#define OMP_KMEANS

double calEuclideanDistance(float *pointA, float *pointB);

void threadingCalCluster(int tId, int N, int K, int num_threads, float *data_points, float **cluster_points, int *num_iterations);

void kmeansClusterOpenMP(int N, int K, int num_threads, float *data_points, float **cluster_points, float **centroids, int *num_iterations);

#endif