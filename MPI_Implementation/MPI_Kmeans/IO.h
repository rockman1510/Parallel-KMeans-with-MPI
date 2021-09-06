#ifndef IO_H
#define IO_H

void readData(const char *filename, int *N, float **data_points);

void writeCluster(const char *filename, int N, float *cluster_points);

void writeCentroids(const char *filename, int K, float *centroids);

#endif