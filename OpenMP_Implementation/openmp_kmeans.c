#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <omp.h>
#include "openmp_kmeans.h"

#define MAX_ITERATORS 50000
#define THREASHOLD 1e-4

float *centroids_global;
float *cluster_points_global;
float detal_global = THREASHOLD + 1;
double current_delta_global_iteration;
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
	while((detal_global > THREASHOLD) && (counter < MAX_ITERATORS)){
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
					// printf("store cluster_points - index %d: %f\t%f\t%f\t%f\n", i, (*cluster_points)[(i * 4)], (*cluster_points)[(i * 4 ) + 1], (*cluster_points)[(i * 4) + 2], (*cluster_points)[(i * 4) + 3]);
				}
			}
			// printf("current_cluster_id[i - start]: %d\n", current_cluster_id[i - start]);

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
				// Calculate mean of a cluster
				centroids_global[((counter + 1) * K + i) * 3] = current_centroid[(i * 3)] / (float)cluster_count[i];
				centroids_global[((counter + 1) * K + i) * 3 + 1] = current_centroid[(i * 3) + 1] / (float)cluster_count[i];
				centroids_global[((counter + 1) * K + i) * 3 + 2] = current_centroid[(i * 3) + 2] / (float)cluster_count[i];
			}
		}

		double current_delta = 0.0;
		for (int i = 0; i < K; i++){
			current_delta += calEuclideanDistance((centroids_global + (counter * K + i) * 3), (centroids_global + ((counter - 1) * K + i) * 3));
		}

// #pragma omp parallel for reduction(+ : sum)
// 		for (int i = 0; i < N; i++){

// 		}
#pragma omp barrier
		{
			if (current_delta > current_delta_global_iteration)
				current_delta_global_iteration = current_delta;
		}

#pragma omp barrier
		counter++;

#pragma omp master
		{
			// printf("counter: %d\n", counter);
			detal_global = current_delta_global_iteration;
			current_delta_global_iteration = 0.0;
			(*num_iterations)++;
			global_counter = counter;
		}
		// printf("======================Thread %d========================\n", tId);
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
		printf("Thread Number: %d created\n", tId);
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