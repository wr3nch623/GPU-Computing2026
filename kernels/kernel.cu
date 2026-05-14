
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

__global__ void COO_FLOPS_CONT(COOStorage* coo, float* vector, float* results, int n, float cont){
   int i = threadIdx.x + blockIdx.x * blockDim.x;
   float res = 0;
   if (i < n){
       //atomicAdd(&results[coo[i].arow], coo[i].aval * vector[coo[i].acol]);
       atomicAdd(&res, coo[i].aval * vector[coo[i].acol]);
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

__global__ void CSRVector_SpVM_NVDA(int* csr_row, int* csr_col, float* csr_val, float* results, float* lineVector, int n){
  extern  __shared__ float vals[];

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int warp_id = i/32;
  int lane = i & 31;

  if(warp_id < n){
      vals[threadIdx.x] = 0;
  for(int j = csr_row[warp_id] + lane; j < csr_row[warp_id+1]; j+=32){
      vals[threadIdx.x] += csr_val[j] * lineVector[csr_col[j]];
  }

  __syncwarp();

  if(lane < 16) vals[threadIdx.x] += vals[threadIdx.x + 16];
  if(lane < 8) vals[threadIdx.x] += vals[threadIdx.x + 8];
  if(lane < 4) vals[threadIdx.x] += vals[threadIdx.x + 4];
  if(lane < 2) vals[threadIdx.x] += vals[threadIdx.x + 2];
  if(lane < 1) vals[threadIdx.x] += vals[threadIdx.x + 1];

  if(lane == 0)
      results[warp_id] = vals[threadIdx.x];

  }
}
