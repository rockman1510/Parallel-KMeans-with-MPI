#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "IO.h"
#include "omp_kmeans.h"

int main(int argc, char const *argv[]){
	if(argc != 6){
		printf("Please input with format: [output file name] [dataset file name] [number of threads] [number of clusters] [cluster file name output] [centroid file name output]");
		return 0;
	}

	const char *dataset_filename = argv[1];
	const int num_threads = atoi(argv[2]);
	const int K = atoi(argv[3]);
	const char *data_points_output_filename = argv[4];
	const char *centroids_output_filename = argv[5];

	int N;
	float *data_points;
	float *cluster_points;
	float *centroids;
	int iterations = 0;
	
	readData(dataset_filename, &N, &data_points);

	double start_time = omp_get_wtime();
	printf("====================Process Started====================\n");
	printf("data_points size: %d\n", N);
	for (int i = 0; i < N; i++){
		printf("i = %d\tdata: %f %f %f\n", i, data_points[(i * 3)], data_points[(i * 3) + 1], data_points[(i * 3) + 2]);
		if(i == 50)
			break;
	}
	kmeansClusterOpenMP(N, K, num_threads, data_points, &cluster_points, &centroids, &iterations);
	double end_time = omp_get_wtime();
	printf("Executed Time: %lfs\n", end_time - start_time);

	writeCluster(data_points_output_filename, N, cluster_points);
	writeCentroids(centroids_output_filename, K, iterations, centroids);

	return 0;
}