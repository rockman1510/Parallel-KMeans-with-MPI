# Parallel-KMeans-with-MPI
This is an example of KMeans Algorithm integrated with MPI4py libaray and OpenMP, which help to run in parallel process.
# Install MPI4py for your OS
pip install mpi4py

# Intall MinGW for your OS (OpenMP implementation):
https://sourceforge.net/projects/mingw-w64/
or
https://sourceforge.net/projects/codeblocks/files/

# Download and install MPI SDK for windows via this link
https://www.microsoft.com/en-us/download/details.aspx?id=57467

# Execute Command:
## For Integrated MPI4py Implementation:
mpiexec -n [processors number] python [python script file name]

## For Integrated OpenMP Implementation:
cd OpenMP_Implementation
ggc main.c openmp_kmeans.c IO.c -fopenmp -o [output file name]
- for example:
ggc main.c openmp_kmeans.c IO.c -fopenmp -o kmeans

[output file name] [dataset file name] [number of threads] [number of clusters] [cluster file name output] [centroid file name output]
- for example:
kmeans kmeans_dataset_1.txt 8 3 cluster_3_8.txt centroid_3_8.txt

## For normal Multi-Thread in Python environment:
python thread_KMeans.py
