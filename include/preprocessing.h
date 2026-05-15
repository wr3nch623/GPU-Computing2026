#include <stdio.h>
#ifndef PRP_H
#define PRP_H

struct COOStorage* matrix_parser(FILE* file, int* rows, int* cols, int* nnz);

bool compareCOO(const COOStorage &a, const COOStorage &b);

bool compareCOOByRow(const COOStorage &a, const COOStorage &b);
void createCSR(COOStorage* coostorage, int nelements, int rows,
               int *csr_row, int *csr_col, float  *csr_val);



#endif
