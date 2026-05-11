#ifndef QQ_MATH_LOADED
#define QQ_MATH_LOADED
void matmult(const float* mat1, const float* mat2, float* output,
             int mat1rows, int m1colsm2rows, int mat2cols,
             bool transA = false, bool transB = false,
             float alpha = 1.0f, float beta = 1.0f);
#endif