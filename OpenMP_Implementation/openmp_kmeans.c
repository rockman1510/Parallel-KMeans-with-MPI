#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include "openmp_kmeans.h"

#define MAX_ITERATORS 1000
#define THREASHOLD 1e-3

float *centroids_global;
float *cluster_points_global;
float distance_global = THREASHOLD + 1;
double distance_per_iteration;
int global_counter = 0;

double calEuclideanDistance(float *point1, float *point2){
	return sqrt(pow(((double)(*(point2 + 0)) - (double)(*(point1 + 0))), 2) + pow(((double)(*(point2 + 1)) - (double)(*(point1 + 1))), 2) + pow(((double)(*(point2 + 2)) - (double)(*(point1 + 2))), 2));
}

void threadingCalCluster(int tId, int N, int K, int num_threads, float *data_points, float **cluster_points, int *num_iterations){
	int length_per_thread = N / num_threads;
	int start = tId * length_per_thread;
	int end = start + length_per_thread;
	if((num_threads - tId) == 1 || end > N){
		end = N;
		length_per_thread = end - start;
	}

	double min_distance, current_distance;
	int *current_cluster_id = (int *)malloc(sizeof(int) * length_per_thread);
	int counter = 0;
	while((distance_global > THREASHOLD) && (counter < MAX_ITERATORS)){
		float *current_centroid = (float *)calloc(K * 3, sizeof(float));
		int *cluster_count = (int *)calloc(K, sizeof(int));
		for (int i = start; i < end; i++){
			min_distance = __DBL_MAX__;
			for (int j = 0; j < K; j++){
				current_distance = calEuclideanDistance((data_points + (i * 3)), (centroids_global + (counter * K + j) * 3));
				if (current_distance < min_distance){
					min_distance = current_distance;
					current_cluster_id[i - start] = j;
					(*cluster_points)[(i * 4)] = data_points[(i * 3)];
					(*cluster_points)[(i * 4) + 1] = data_points[(i * 3) + 1];
					(*cluster_points)[(i * 4) + 2] = data_points[(i * 3) + 2];
					(*cluster_points)[(i * 4) + 3] = j;
				}
			}

			// Sum all the point belong to a luster
			cluster_count[current_cluster_id[i - start]]++;
			current_centroid[(current_cluster_id[i - start] * 3)] += data_points[(i * 3)];
			current_centroid[(current_cluster_id[i - start] * 3) + 1] += data_points[(i * 3) + 1];
			current_centroid[(current_cluster_id[i - start] * 3) + 2] += data_points[(i * 3) + 2];
		}

#pragma omp critical
		{
			for (int i = 0; i < K; i++){
				if (cluster_count[i] == 0){
					continue;
				}
				// Calculate mean of a cluster and update it
				centroids_global[((counter + 1) * K + i) * 3] = current_centroid[(i * 3)] / (float)cluster_count[i];
				centroids_global[((counter + 1) * K + i) * 3 + 1] = current_centroid[(i * 3) + 1] / (float)cluster_count[i];
				centroids_global[((counter + 1) * K + i) * 3 + 2] = current_centroid[(i * 3) + 2] / (float)cluster_count[i];
			}
		}
		
		// Find distance value after each iteration in all the threads
		double current_distance = 0.0;
		for (int i = 0; i < K; i++){
			current_distance += calEuclideanDistance((centroids_global + (counter * K + i) * 3), (centroids_global + ((counter - 1) * K + i) * 3));
		}

#pragma omp barrier
		{
			if (current_distance > distance_per_iteration)
				distance_per_iteration = current_distance;
		}

#pragma omp barrier
		counter++;

#pragma omp master
		{
			// printf("counter: %d\n", counter);
			distance_global = distance_per_iteration;
			distance_per_iteration = 0.0;
			(*num_iterations)++;
			global_counter = counter;
		}
	}
}


void kmeansClusterOpenMP(int N, int K, int num_threads, float *data_points, float **cluster_points, float **centroids, int *num_iterations){
	*cluster_points = (float *)malloc(sizeof(float) * N * 4);
	centroids_global = (float *)calloc(MAX_ITERATORS * K * 3, sizeof(float));
	for(int i = 0; i < K; i++){
		centroids_global[(i * 3)] = data_points[(i * 3)];
		centroids_global[(i * 3) + 1] = data_points[(i * 3) + 1];
		centroids_global[(i * 3) + 2] = data_points[(i * 3) + 2];
		printf("centroid %d: %f\t%f\t%f\n", i, centroids_global[(i * 3)], centroids_global[(i * 3) + 1], centroids_global[(i * 3) + 2]);
	}
	omp_set_num_threads(num_threads);

#pragma omp parallel
	{
		int tId = omp_get_thread_num();
		// printf("Thread Number: %d created\n", tId);
		threadingCalCluster(tId, N, K, num_threads, data_points, cluster_points, num_iterations);
	}

	int centroids_size = (*num_iterations + 1) * K * 3;
	*centroids = (float *)calloc(centroids_size, sizeof(float));
	for (int i = 0; i < centroids_size; i++){
		(*centroids)[i] = centroids_global[i];
	}

	printf("====================Process Completed====================\n");
	printf("Number of Iterations: %d\n", global_counter);
	for (int i = 0; i < K; i++){
		printf("Index: %d\tFinal Centroids:\t(%f, %f, %f)\n", i, *(*centroids + ((*num_iterations) * K) + (i * 3)), *(*centroids + ((*num_iterations) * K) + (i * 3) + 1), *(*centroids + ((*num_iterations) * K) + (i * 3) + 2));
	}
}