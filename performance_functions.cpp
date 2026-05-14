#include <iostream>
#include <stdlib.h>

// 1.e6 because time is measured in milliseconds, so i need to account for that
float bandwidthCOOTheoretical(float nnz, float timeElapsed){
    return (nnz * (sizeof(float) + sizeof(int) + sizeof(float) + sizeof(int)))/timeElapsed/1.e6;
}


float bandwidthCSRTheoretical(float rows, float nnz, float timeElapsed){

    return (nnz * ((sizeof(float) + sizeof(int) +
                    sizeof(float))) + (rows * 2 * sizeof(float))) / timeElapsed / 1.e6;

}

float gflops(float flops, float timeElapsed){
    return flops/timeElapsed/1.e6;
}
