#ifndef OPENMP_KMEANS
#define OPENMP_KMEANS

double calEuclideanDistance(float *point1, float *point2);

void threadingCalCluster(int tId, int N, int K, int num_threads, float *data_points, float **cluster_points, int *num_iterations);

void kmeansClusterOpenMP(int N, int K, int num_threads, float *data_points, float **cluster_points, float **centroids, int *num_iterations);

#endif