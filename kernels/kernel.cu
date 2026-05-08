
// --------------------------
// - COO
// --------------------------
__global__ void COO_SpVM_NVDA(COOStorage* coo, float* vector, float* results, int n){
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  if (i < n){
      //results[coo[i].arow] += coo[i].aval * vector[coo[i].acol];
      atomicAdd(&results[coo[i].arow], coo[i].aval * vector[coo[i].acol]);
  }
}

__global__ void COO_FLOPS_CONT(float cont, int n){
   int i = threadIdx.x + blockIdx.x * blockDim.x;

   if (i < n){
       //atomicAdd(&results[coo[i].arow], coo[i].aval * vector[coo[i].acol]);
       atomicAdd(&cont, 1);
   }

}



// --------------------------
// - CSR
// --------------------------
__global__ void CSR_SpVM_NVDA(int* csr_row, int* csr_col, float* csr_val, float* results, float* lineVector, int n){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  if(i < n){
      float sum = 0;
  for(int j = csr_row[i]; j < csr_row[i+1]; j++){
      sum += csr_val[j] * lineVector[csr_col[j]];
  }
  results[i] = sum;
  }
}

