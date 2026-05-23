#ifndef QQ_MATH_CPP
#define QQ_MATH_CPP
#include "math.h"
#ifdef QQ_BLAS_NPU
#include "npublas.h"
#endif
#ifdef QQ_BLAS_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#ifdef QQ_BLAS_CLBLAST
#include <clblast_c.h>
#include <CL/cl.h>
#endif
#ifdef QQ_BLAS_GENERIC
#include <cblas.h>
#endif
void matmult(const float* matrix_a, const float* matrix_b, float* result,
             int rows_a, int inner_dimension, int cols_b,
             bool transpose_a, bool transpose_b,
             float alpha, float beta) {
#ifdef QQ_BLAS_NPU
    if (!transpose_a && !transpose_b && alpha == 1.0f && beta == 0.0f) {
        if (npublas::npu_available()) {
            int err = npublas::matmul_8bit(matrix_a, matrix_b, result, rows_a, inner_dimension, cols_b);
            if (err == 0) return;
        }
    }
#endif
#ifdef QQ_BLAS_CUBLAS
    static cublasHandle_t cublas_handle = nullptr;
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
    int output_rows = transpose_a ? inner_dimension : rows_a;
    int output_cols = transpose_b ? inner_dimension : cols_b;
    int inner_dimension_a = transpose_a ? rows_a : inner_dimension;
    int inner_dimension_b = transpose_b ? cols_b : inner_dimension;
    if (inner_dimension_a != inner_dimension_b) goto naive_fallback;
    int leading_dimension_a = transpose_a ? rows_a : inner_dimension;
    int leading_dimension_b = transpose_b ? cols_b : inner_dimension;
    int leading_dimension_c = output_cols;
    float *d_a = nullptr, *d_b = nullptr, *d_result = nullptr;
    cudaMalloc(&d_a, rows_a * inner_dimension * sizeof(float));
    cudaMalloc(&d_b, inner_dimension * cols_b * sizeof(float));
    cudaMalloc(&d_result, output_rows * output_cols * sizeof(float));
    cudaMemcpy(d_a, matrix_a, rows_a * inner_dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, matrix_b, inner_dimension * cols_b * sizeof(float), cudaMemcpyHostToDevice);
    if (beta != 0.0f) {
        cudaMemcpy(d_result, result, output_rows * output_cols * sizeof(float), cudaMemcpyHostToDevice);
    }
    cublasSgemm(cublas_handle,
                transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                output_rows, output_cols, inner_dimension_a,
                &alpha,
                d_a, leading_dimension_a,
                d_b, leading_dimension_b,
                &beta,
                d_result, leading_dimension_c);
    cudaMemcpy(result, d_result, output_rows * output_cols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);
    return;
naive_fallback:
#endif
#ifdef QQ_BLAS_CLBLAST
    static cl_platform_id platform = nullptr;
    static cl_device_id device = nullptr;
    static cl_context context = nullptr;
    static cl_command_queue queue = nullptr;
    static bool cl_initialized = false;
    if (!cl_initialized) {
        clGetPlatformIDs(1, &platform, nullptr);
        clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
        context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
        queue = clCreateCommandQueue(context, device, 0, nullptr);
        cl_initialized = true;
    }
    int output_rows = transpose_a ? inner_dimension : rows_a;
    int output_cols = transpose_b ? inner_dimension : cols_b;
    int inner_dimension_a = transpose_a ? rows_a : inner_dimension;
    int inner_dimension_b = transpose_b ? cols_b : inner_dimension;
    if (inner_dimension_a != inner_dimension_b) goto naive_fallback;
    int leading_dimension_a = transpose_a ? rows_a : inner_dimension;
    int leading_dimension_b = transpose_b ? cols_b : inner_dimension;
    int leading_dimension_c = output_cols;
    cl_int err;
    cl_mem d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, rows_a * inner_dimension * sizeof(float), nullptr, &err);
    cl_mem d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, inner_dimension * cols_b * sizeof(float), nullptr, &err);
    cl_mem d_result = clCreateBuffer(context, CL_MEM_READ_WRITE, output_rows * output_cols * sizeof(float), nullptr, &err);
    clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, rows_a * inner_dimension * sizeof(float), matrix_a, 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, inner_dimension * cols_b * sizeof(float), matrix_b, 0, nullptr, nullptr);
    if (beta != 0.0f) {
        clEnqueueWriteBuffer(queue, d_result, CL_TRUE, 0, output_rows * output_cols * sizeof(float), result, 0, nullptr, nullptr);
    }
    CLBlastSgemm(CLBlastLayoutRowMajor,
                 transpose_a ? CLBlastTransposeYes : CLBlastTransposeNo,
                 transpose_b ? CLBlastTransposeYes : CLBlastTransposeNo,
                 output_rows, output_cols, inner_dimension_a,
                 alpha,
                 d_a, 0, leading_dimension_a,
                 d_b, 0, leading_dimension_b,
                 beta,
                 d_result, 0, leading_dimension_c,
                 &queue, nullptr);
    clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, output_rows * output_cols * sizeof(float), result, 0, nullptr, nullptr);
    clReleaseMemObject(d_a); clReleaseMemObject(d_b); clReleaseMemObject(d_result);
    return;
naive_fallback:
#endif
#ifdef QQ_BLAS_GENERIC
    int output_rows = transpose_a ? inner_dimension : rows_a;
    int output_cols = transpose_b ? inner_dimension : cols_b;
    int inner_dimension_a = transpose_a ? rows_a : inner_dimension;
    int inner_dimension_b = transpose_b ? cols_b : inner_dimension;
    if (inner_dimension_a != inner_dimension_b) {
        goto naive_fallback;
    }
    cblas_sgemm(CblasRowMajor,
                transpose_a ? CblasTrans : CblasNoTrans,
                transpose_b ? CblasTrans : CblasNoTrans,
                output_rows, output_cols, inner_dimension_a,
                alpha,
                matrix_a, transpose_a ? rows_a : inner_dimension,
                matrix_b, transpose_b ? cols_b : inner_dimension,
                beta,
                result, output_cols);
    return;
naive_fallback:
#endif
    int opA_rows = transpose_a ? inner_dimension : rows_a;
    int opA_cols = transpose_a ? rows_a : inner_dimension;
    int opB_rows = transpose_b ? cols_b : inner_dimension;
    int opB_cols = transpose_b ? inner_dimension : cols_b;
    int output_rows = opA_rows;
    int output_cols = opB_cols;
    for (int row = 0; row < output_rows; ++row) {
        for (int col = 0; col < output_cols; ++col) {
            float sum = 0.0f;
            for (int inner = 0; inner < opA_cols; ++inner) {
                float a_val = transpose_a ? matrix_a[inner * inner_dimension + row] : matrix_a[row * inner_dimension + inner];
                float b_val = transpose_b ? matrix_b[col * inner_dimension + inner] : matrix_b[inner * cols_b + col];
                sum += a_val * b_val;
            }
            result[row * output_cols + col] = beta * result[row * output_cols + col] + alpha * sum;
        }
    }
}
#endif