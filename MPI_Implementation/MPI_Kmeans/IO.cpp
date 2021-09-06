#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include "IO.h"

void readData(const char* filename, int* N, float** data_points) {

	FILE* f = fopen(filename, "r");
	if (f == NULL) {
		return;
	}
	fscanf(f, "%d", N);
	//printf("Number of data points in the dataset: %d\n", *N);
	*data_points = (float*)malloc(((*N) * 3) * sizeof(float));

	for (int i = 0; i < (*N) * 3; i++) {
		fscanf(f, "%f", (*data_points + i));
	}
	fclose(f);
}

void writeCluster(const char* filename, int N, float* cluster_points) {

	FILE* f = fopen(filename, "w");

	for (int i = 0; i < N; i++) {
		fprintf(f, "%f %f %f %d\n",
			*(cluster_points + (i * 4)), *(cluster_points + (i * 4) + 1),
			*(cluster_points + (i * 4) + 2), (int)*(cluster_points + (i * 4) + 3));
	}
	fclose(f);
}

void writeCentroids(const char* filename, int K, float* centroids) {
	FILE* f = fopen(filename, "w");
		for (int i = 0; i < K; i++) {
			fprintf(f, "%f %f %f\n", *(centroids + (i * K)),
				*(centroids + (i * K) + 1),
				*(centroids + (i * K) + 2));
		}
		fprintf(f, "\n");
	fclose(f);
}