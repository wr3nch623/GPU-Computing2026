#include <iostream>
#include <stdlib.h>


float flops(float nnz, float timeElapsed){
    return (2*nnz)/timeElapsed;
}
