## Parallel-KMeans-with-MPI
This is an example of KMeans Algorithm integrated with MPI4py libaray and OpenMP, which help to run in parallel process.
## Install MPI4py for your OS
_pip install mpi4py_

## Intall MinGW for your OS (OpenMP implementation):
https://sourceforge.net/projects/mingw-w64/
or
https://sourceforge.net/projects/codeblocks/files/

## Download and install MPI SDK for windows via this link
https://www.microsoft.com/en-us/download/details.aspx?id=57467

## Execute Command:
### For Integrated MPI4py Implementation:
_mpiexec -n [processors number] python [python script file name]_ _[number of data] [number of clusters]_

### For Integrated OpenMP Implementation:
_cd OpenMP_Implementation_

_ggc main.c openmp_kmeans.c IO.c -fopenmp -o [output file name]_
- for example:
_ggc main.c openmp_kmeans.c IO.c -fopenmp -o kmeans_

_[output file name] [dataset file name] [number of threads] [number of clusters] [number of data] [cluster file name output] [centroid file name output]_
- for example:
_kmeans kmeans_dataset_1.txt 8 3 200000 cluster_3_8.txt centroid_3_8.txt_

### For normal Multi-Thread in Python environment:
_python thread_KMeans.py_
