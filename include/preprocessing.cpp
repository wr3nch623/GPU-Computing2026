#include "preprocessing.h"
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <vector>
#include <algorithm>
#include <tuple>
#include <sys/time.h>

#include "types.h"

struct COOStorage* createCOO(int nelements){
    struct COOStorage* storage = NULL;
    storage = (COOStorage*)malloc(sizeof(struct COOStorage) * nelements);

    for(int i = 0; i < nelements; i++){
        storage[i].arow = 0;
        storage[i].acol = 0;
        storage[i].aval = 0;
    }

    return storage;

}

// Functions that compare rows and columns to sort them in crescent order column or row based.
bool compareCOO(const COOStorage &a, const COOStorage &b){
    return std::tie(a.acol, a.arow) < std::tie(b.acol, b.arow);
}


bool compareCOOByRow(const COOStorage &a, const COOStorage &b){
    return std::tie(a.arow, a.acol) < std::tie(b.arow, b.acol);
}



// TODO: When csr implemented fix this function
void createCSR(COOStorage* coostorage, int nelements, int rows,
               int *csr_row, int *csr_col, float  *csr_val){


    // sort COO by rows NOTE: Already do this in main, resort is useless
    //std::sort(coostorage, coostorage + nelements, compareCOOByRow);
    memset(csr_row, 0, (rows + 1) * sizeof(int));
    for (int i = 0; i < nelements; i++) {
        csr_row[coostorage[i].arow + 1]++;
    }

    for(int i = 0; i < rows; i++){
        csr_row[i+1] += csr_row[i];
    }

    int *temp = (int *)malloc(rows * sizeof(int));
    for (int i = 0; i < rows; ++i)
        temp[i] = csr_row[i];

    for (int i = 0; i < nelements; ++i) {
        int row = coostorage[i].arow;
        int idx = temp[row]++;
        csr_col[idx] = coostorage[i].acol;
        csr_val[idx] = coostorage[i].aval;
    }

    free(temp);
}


struct COOStorage* matrix_parser(FILE* file, int* rows, int* cols, int* nnz){
    char tempString[10000];
    bool first = false;
    float arr[3];
    struct COOStorage* coomatrix;
    int i = 0;

    // Read file
    while(fgets(tempString, 10000, file)){
        int k, h;

        if(!(tempString[0] == '%')){
            // Read and populate the array
            int matches = sscanf(tempString, "%f %f %f", &arr[0], &arr[1], &arr[2]);

            // Sometimes in the files there is no third number, in this case it is a 1.
            if(matches == 2){
                arr[2] = 1;
            }

            //printf("%f, %f, %f\n",  (int)arr[0], arr[1], arr[2]);

            // Check if the array already was created
            if(!first){
                *rows = (int)arr[0];
                *cols = (int)arr[1];
                *nnz = (int)arr[2];

                coomatrix = createCOO(*nnz);


                first = true;
                continue;
            }

            coomatrix[i].arow = arr[0] - 1;
            coomatrix[i].acol = arr[1] - 1;

            // If there are only two numbers for each record (e.g. row, col) it is treated as if there were three
            // but the third is a 1 (e.g row, col, 1) since those formats only save non zero information
            if(matches == 2)
                coomatrix[i].aval = 1;
            // Otherwise do whatever and save it, clearly it has a problem with precision
            // floats only save up until a certain point and after 1e-8 it does not, but
            // its fine i feel... Right?
            else {
                coomatrix[i].aval = (float)arr[2];
                //printf("%f, %f\n", arr[2], coomatrix[i].aval);
            }
            i++;

        }
    }

    return coomatrix;

}
