
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



// --------------------------
// - CSR
// --------------------------
__global__ void CSR_SpVM_NVDA(int* csr_row, int* csr_col, float* csr_val, float* results, int n){
  int i = threadIdx.x + blockIdx.x * blockDim.x;


}
