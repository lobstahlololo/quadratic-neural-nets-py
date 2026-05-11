#include "math.h"
#ifdef QQ_BLAS_GENERIC
#include <cblas.h>
#endif
void matmult(const float* mat1, const float* mat2, float* output,
             int mat1rows, int m1colsm2rows, int mat2cols,
             bool transA, bool transB,
             float alpha, float beta) {
#ifdef QQ_BLAS_GENERIC
    int m = transA ? m1colsm2rows : mat1rows;
    int n = transB ? m1colsm2rows : mat2cols;
    int k = transA ? mat1rows : m1colsm2rows;
    int k_b = transB ? mat2cols : m1colsm2rows;
    if (k != k_b) {
        goto naive_fallback;
    }
    cblas_sgemm(CblasRowMajor,
                transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans,
                m, n, k,
                alpha,
                mat1, transA ? m1colsm2rows : mat1rows,
                mat2, transB ? mat2cols : m1colsm2rows,
                beta,
                output, n);
    return;
naive_fallback:
#endif
    int a_rows = transA ? m1colsm2rows : mat1rows;
    int a_cols = transA ? mat1rows : m1colsm2rows;
    int b_rows = transB ? mat2cols : m1colsm2rows;
    int b_cols = transB ? m1colsm2rows : mat2cols;
    int out_rows = a_rows;
    int out_cols = b_cols;
    for (int i = 0; i < out_rows; ++i) {
        for (int j = 0; j < out_cols; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < a_cols; ++k) {
                float a_val;
                if (transA) {
                    a_val = mat1[k * m1colsm2rows + i];
                } else {
                    a_val = mat1[i * a_cols + k];
                }
                float b_val;
                if (transB) {
                    b_val = mat2[j * m1colsm2rows + k];
                } else {
                    b_val = mat2[k * b_cols + j];
                }
                sum += a_val * b_val;
            }
            output[i * out_cols + j] = beta * output[i * out_cols + j] + alpha * sum;
        }
    }
}