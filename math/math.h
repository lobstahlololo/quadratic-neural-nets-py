#ifndef QQ_MATH_LOADED
#define QQ_MATH_LOADED

void preload();
typedef void (*MatmulInnerHook)(const float* a, const float* b, float* c, int rows_a, int inner_dimension, int cols_b, int chunk_idx, int total_chunks);
void matmult(const float* matrix_a, const float* matrix_b, float* result,
             int rows_a, int inner_dimension, int cols_b,
             bool transpose_a = false, bool transpose_b = false,
             float alpha = 1.0f, float beta = 1.0f,
             int split_into = 1, MatmulInnerHook inner_hook = nullptr);
#endif