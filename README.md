# GPU-Computing2026
This repo is for the GPU Computing project development. There will be implementations for Sparse Matrix Multiplication and optimization of the algorithm that will run on an NVidia GPU.


Implemented:
- COO
  - COO OpenMP computation
  - COO NVIDIA
- CSR
  - CSR OpenMP
  - CSR Vector
  
- redo time library

Missing:
- COO
  - cuSparse 
- CSR
  - cuSparse
  - Other stride accesses or warp parallelism implementation
  
  - Fix code
  - Write documentation and compilation guide 
  - time measurement on everything
  - compute performance metrics
  - data logging and generation
  
Possible optimization:
    - Stride 
    - async memory copy
    - pipeline everything to keep computation authority on GPU to better benchmark performance
    - directly using warps
  
- Bandwidth computation
- flops computation
- other things that i am missing atm
