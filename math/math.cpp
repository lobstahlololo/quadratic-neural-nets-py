

#include "math.h"

void matmult(const float* mat1, const float* mat2, float* output, 
             int mat1rows, int m1colsm2rows, int mat2cols) {
    float* current = output;
    for (int i = 0; i < mat1rows; ++i) {
        current = output + i * mat2cols;
        for (int k = 0; k < m1colsm2rows; ++k) {
            float mat1_ik = *(mat1 + i * m1colsm2rows + k);
            const float* mat2_k = mat2 + k * mat2cols;
            for (int j = 0; j < mat2cols; ++j) {
                *(current + j) += mat1_ik * *(mat2_k + j);
            }
        }
    }
}