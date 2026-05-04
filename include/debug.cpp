
#include "types.cpp"
#include <stdio.h>

// TODO: put this in a function
void yetAnotherDebugFunc(struct COOStorage *coomatrix, int nnz){
    for(int i = 0; i < nnz; i++){
        printf("%f ", coomatrix[i].aval);
        // FLUSH cause otherwise in debug it will not print, and we all know which is the clearly superior way of debugging
        fflush(stdout);
    }
}


void debugPrintCOOMatric(struct COOStorage *coomatrix, int rows, int cols, int nnz){
    int counter = 0;
    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            if(coomatrix[counter].arow == i && coomatrix[counter].acol == j){
                printf("%f ", coomatrix[counter].aval);
                counter++;
            }
            else
                printf("0 ");
        }
        printf("\n");
    }
}


// Debug print just to check it works
void print_csr_matrix(int *csr_row, int *csr_col, float *csr_val, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        int entry = csr_row[i];
        for (int j = 0; j < cols; j++) {
            if (entry < csr_row[i + 1] && csr_col[entry] == j) {
                printf("%6.1f ", csr_val[entry++]);
            } else {
                printf("%6.1f ", 0.0f);
            }
        }
    printf("\n");
    }
}
