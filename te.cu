
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>


int main(int argc, char *argv[]) {



    float* A; float* B; float* C;
    int dim = 5000 * sizeof(float);
    struct timeval inittime;
    struct timeval endtime;


    A = (float*)malloc(dim);
    B = (float*)malloc(dim);
    C = (float*)malloc(dim);

    for(int i = 0; i < 5000; i++){
        A[i] = 10;
        B[i] = 20;
    }


   float* d_A, *d_B, *d_C;

   gettimeofday(&inittime, NULL);
    cudaMalloc((float**)d_A, dim);
    cudaMalloc((float**)d_B, dim);
    cudaMalloc((float**)d_C, dim);

    cudaMemcpy(A, d_A, dim, cudaMemcpyHostToDevice);
    cudaMemcpy(B, d_B, dim, cudaMemcpyHostToDevice);
    cudaMemcpy(C, d_C, dim, cudaMemcpyHostToDevice);

    gettimeofday(&endtime, NULL);

    printf();








    return 0;
}
