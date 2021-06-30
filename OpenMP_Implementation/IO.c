#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#include "IO.h"

void readData(const char *filename, int *N, float **data_points){
	
	FILE *f = fopen(filename, "r");
	fscanf(f, "%d",N);
	printf("Number of data points in the dataset: %d\n", *N);
	*data_points = (float *)malloc(((*N) * 3) * sizeof(float));

	for (int i = 0; i < (*N) * 3; i++){
		int temp;
		fscanf(f, "%d", &temp);
		*(*data_points + i) = temp;
	}
	fclose(f);
}

void writeCluster(const char *filename, int N, float *cluster_points){

	FILE *f = fopen(filename, "w");

	for (int i = 0; i < N; i++){
		fprintf(f, "%f %f %f %f\n", 
			*(cluster_points + (i * 4)), *(cluster_points + (i * 4) + 1), 
			*(cluster_points + (i * 4) + 2), *(cluster_points + (i * 4) + 3));
	}
	fclose(f);
}

void writeCentroids(const char *filename, int K, int num_iterations, float *centroids){
	FILE *f = fopen(filename, "w");
	for (int i = 0; i < num_iterations; i++){
		for (int j = 0; j < K; j++){
			fprintf(f, "%f %f %f, ", *(centroids + (i * K) + (j * 3)), 
				*(centroids + (i * K) + (j * 3) + 1), 
				*(centroids + (i * K) + (j * 3) + 2));
		}
	}
	fclose(f);
}