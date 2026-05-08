# GPU-Computing2026
This repo is for the GPU Computing project development. There will be implementations for Sparse Matrix Multiplication and optimization of the algorithm that will run on an NVidia GPU.


Implemented:
- COO
  - COO OpenMP computation
  - COO NVIDIA
- CSR
  - CSR OpenMP
  - NVIDIA

Missing:
- COO
  - cuSparse 
- CSR
  - cuSparse
  - CSR Vector
  - Other stride accesses or warp parallelism implementation
  
  - Fix code
  - Write documentation and compilation guide 

  
Possible optimization:
    - Stride 
    - async memory copy
    - directly using warps
  
- Bandwidth computation
- flops computation
- other things that i am missing atm
