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
### For Integrated MPI Implementation:
_cd MPI_Implementation\Debug_

_mpiexec -n [processors number] [output file name]_ _[number of clusters] [dataset file name]_
- for example:
_mpiexec -n 2 MPI_Kmeans_ _3 dataset-100000.txt_
### For Integrated OpenMP Implementation:
_cd OpenMP_Implementation_

_ggc Kmeans_OpenMP.c -fopenmp -o [output file name]_
- for example:
_ggc Kmeans_OpenMP.c -fopenmp -o kmeans_

_[output file name] [number of threads] [number of clusters] [dataset file name]_
- for example:
_kmeans 2 3 dataset-100000.txt_

### For Sequential Kmeans in C:
_ggc sequential_kmeans.c -o [output file name]_

_ggc sequential_kmeans.c -o [output file name]_
- for example:
_ggc sequential_kmeans.c -o kmeans_sequential_

_[output file name] [number of clusters] [dataset file name]_
- for example:
_kmeans_sequential 3 dataset-100000.txt_
