#ifndef PFR_H
#define PFR_H
// 1.e6 because time is measured in milliseconds, so i need to account for that
float bandwidthCOOTheoretical(float nnz, float timeElapsed);

float bandwidthCSRTheoretical(float rows, float nnz, float timeElapsed);
float gflops(float flops, float timeElapsed);

float rmse(float*, float*, int);

#endif
