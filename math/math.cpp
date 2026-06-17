#ifndef QQ_MATH_CPP
#define QQ_MATH_CPP
#include <cstdlib>
#include <iostream>
#include <chrono>
#include "math.h"
#include "fixedpoint.h"

#ifdef QQ_BLAS_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif
#ifdef QQ_BLAS_CLBLAST
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include <clblast_c.h>
#include <dlfcn.h>
#endif
#ifdef QQ_BLAS_NPU
#include "litert_matmul/tflite_manager.h"
#endif
#ifdef QQ_BLAS_GENERIC
#include <cblas.h>
#endif
/*#ifdef QQ_BLAS_INTEIGHT

#include "fixedpoint.h"
#endif
*/

void preload() {
}

void matmult_impl(const float* matrix_a, const float* matrix_b, float* result,
             int rows_a, int inner_dimension, int cols_b,
             bool transpose_a, bool transpose_b,
             float alpha, float beta,
             const float* next_weights = nullptr, int next_weights_size = 0) {
    if (rows_a == 0 || cols_b == 0 || inner_dimension == 0) return;
    const float* local_a = matrix_a;
    const float* local_b = matrix_b;
#ifdef QQ_BLAS_INTEIGHT
    int block_size = inner_dimension;
    FixedPointBlock<int8_t> block_a(rows_a, inner_dimension, block_size, true);
    block_a.fit_exponent(matrix_a);
    block_a.floats_to_mantissa(matrix_a, 0, 0, 4);

    FixedPointBlock<int8_t> block_b(inner_dimension, cols_b, block_size, false);
    block_b.fit_exponent(matrix_b);
    block_b.floats_to_mantissa(matrix_b, 0, 0, 4);

    FixedPointBlock<int64_t> block_c(rows_a, cols_b, 1, true);
    int num_blocks_k = (inner_dimension + block_size - 1) / block_size;

    for (int i = 0; i < rows_a; ++i) {
        for (int j = 0; j < cols_b; ++j) {
            int64_t accumulator = 0;
            if (!transpose_a && !transpose_b) {
                for (int k = 0; k < inner_dimension; ++k) {
                    accumulator += (int64_t)block_a.mantissa[i * inner_dimension + k] * block_b.mantissa[k * cols_b + j];
                }
            } else if (transpose_a && !transpose_b) {
                for (int k = 0; k < inner_dimension; ++k) {
                    accumulator += (int64_t)block_a.mantissa[k * rows_a + i] * block_b.mantissa[k * cols_b + j];
                }
            } else if (!transpose_a && transpose_b) {
                for (int k = 0; k < inner_dimension; ++k) {
                    accumulator += (int64_t)block_a.mantissa[i * inner_dimension + k] * block_b.mantissa[j * inner_dimension + k];
                }
            } else {
                for (int k = 0; k < inner_dimension; ++k) {
                    accumulator += (int64_t)block_a.mantissa[k * rows_a + i] * block_b.mantissa[j * inner_dimension + k];
                }
            }
            block_c.mantissa[i * cols_b + j] = accumulator;
            int32_t total_exp = 0;
            for (int b = 0; b < num_blocks_k; ++b) {
                total_exp += block_a.exponents[i * num_blocks_k + b] + block_b.exponents[b * cols_b + j];
            }
            block_c.exponents[i * cols_b + j] = total_exp;
        }
    }

    block_c.mantissa_to_floats_product(result, 4);

    for (int i = 0; i < rows_a * cols_b; ++i) {
        result[i] = (beta == 0.0f ? 0.0f : beta * result[i]) + alpha * result[i];
    }
    return;
#endif
#ifdef QQ_BLAS_NPU
    matmul_int8(matrix_a, matrix_b, result, rows_a, inner_dimension, cols_b, transpose_a, transpose_b);
    for (int i = 0; i < rows_a * cols_b; ++i) {
        result[i] = (beta == 0.0f ? 0.0f : beta * result[i]) + alpha * result[i];
    }
    return;
#endif
#ifdef QQ_BLAS_CUBLAS
    thread_local cublasHandle_t cublas_handle = nullptr;
    if (!cublas_handle) {
        cublasCreate(&cublas_handle);
    }
    float *d_a = nullptr, *d_b = nullptr, *d_result = nullptr;
    cudaMalloc(&d_a, rows_a * inner_dimension * sizeof(float));
    cudaMalloc(&d_b, inner_dimension * cols_b * sizeof(float));
    cudaMalloc(&d_result, rows_a * cols_b * sizeof(float));
    cudaMemcpy(d_a, local_a, rows_a * inner_dimension * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, local_b, inner_dimension * cols_b * sizeof(float), cudaMemcpyHostToDevice);
    if (beta != 0.0f) {
        cudaMemcpy(d_result, result, rows_a * cols_b * sizeof(float), cudaMemcpyHostToDevice);
    }
    cublasSgemm(cublas_handle,
                transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N,
                transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                cols_b, rows_a, inner_dimension,
                &alpha,
                d_b, transpose_b ? inner_dimension : cols_b,
                d_a, transpose_a ? rows_a : inner_dimension,
                &beta,
                d_result, cols_b);
    cudaMemcpy(result, d_result, rows_a * cols_b * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_result);
    return;
#endif
#ifdef QQ_BLAS_CLBLAST
    struct CLSharedContext {
        cl_platform_id platform = nullptr;
        cl_device_id device = nullptr;
        cl_context context = nullptr;
        bool failed = true;

        CLSharedContext() {
            cl_uint num_platforms = 0;
            if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) return;
            cl_platform_id platforms[16];
            clGetPlatformIDs(num_platforms < 16 ? num_platforms : 16, platforms, nullptr);
            for (cl_uint i = 0; i < num_platforms && i < 16; ++i) {
                cl_device_id devices[16];
                cl_uint num_devices = 0;
                if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 16, devices, &num_devices) == CL_SUCCESS && num_devices > 0) {
                    device = devices[0];
                    platform = platforms[i];
                    break;
                }
            }
            if (!platform || !device) {
                for (cl_uint i = 0; i < num_platforms && i < 16; ++i) {
                    cl_device_id devices[16];
                    cl_uint num_devices = 0;
                    if (clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 16, devices, &num_devices) == CL_SUCCESS && num_devices > 0) {
                        device = devices[0];
                        platform = platforms[i];
                        break;
                    }
                }
            }
            if (!platform || !device) return;
            
            cl_context_properties properties[] = {
                CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
                0
            };
            cl_int err;
            context = clCreateContext(properties, 1, &device, nullptr, nullptr, &err);
            if (err != CL_SUCCESS || !context) {
                context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
            }
            if (err == CL_SUCCESS && context) {
                failed = false;
            } else {
                static bool p = false; if (!p) { std::cout << "[CL] Context failed with error: " << err << "\n"; p = true; }
            }
        }
    };

    static CLSharedContext shared_ctx;
    if (shared_ctx.failed) {
        static bool p = false; if (!p) { std::cout << "[CL] Context init failed\n"; p = true; }
        goto naive_fallback_cl;
    }

    thread_local cl_command_queue queue = nullptr;
    if (!queue) {
        cl_int err;
        queue = clCreateCommandQueue(shared_ctx.context, shared_ctx.device, 0, &err);
        if (err != CL_SUCCESS || !queue) {
            static bool p = false; if (!p) { std::cout << "[CL] Queue failed: " << err << "\n"; p = true; }
            goto naive_fallback_cl;
        }
    }

    {
        cl_context context = shared_ctx.context;
        cl_int err;
        
        struct OpenCLThreadBufferCache {
            cl_mem device_buffer_a = nullptr;
            size_t capacity_a = 0;
            cl_mem device_buffer_b = nullptr;
            size_t capacity_b = 0;
            cl_mem device_buffer_result = nullptr;
            size_t capacity_result = 0;

            ~OpenCLThreadBufferCache() {
                if (device_buffer_a) {
                    clReleaseMemObject(device_buffer_a);
                }
                if (device_buffer_b) {
                    clReleaseMemObject(device_buffer_b);
                }
                if (device_buffer_result) {
                    clReleaseMemObject(device_buffer_result);
                }
            }
        };

        static thread_local OpenCLThreadBufferCache cache;

        size_t needed_a = rows_a * inner_dimension * sizeof(float);
        if (cache.capacity_a < needed_a) {
            if (cache.device_buffer_a) {
                clReleaseMemObject(cache.device_buffer_a);
            }
            cache.device_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, needed_a, nullptr, &err);
            if (err != CL_SUCCESS || !cache.device_buffer_a) {
                cache.device_buffer_a = nullptr;
                cache.capacity_a = 0;
                goto naive_fallback_cl;
            }
            cache.capacity_a = needed_a;
        }

        size_t needed_b = inner_dimension * cols_b * sizeof(float);
        if (cache.capacity_b < needed_b) {
            if (cache.device_buffer_b) {
                clReleaseMemObject(cache.device_buffer_b);
            }
            cache.device_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, needed_b, nullptr, &err);
            if (err != CL_SUCCESS || !cache.device_buffer_b) {
                cache.device_buffer_b = nullptr;
                cache.capacity_b = 0;
                goto naive_fallback_cl;
            }
            cache.capacity_b = needed_b;
        }

        size_t needed_result = rows_a * cols_b * sizeof(float);
        if (cache.capacity_result < needed_result) {
            if (cache.device_buffer_result) {
                clReleaseMemObject(cache.device_buffer_result);
            }
            cache.device_buffer_result = clCreateBuffer(context, CL_MEM_READ_WRITE, needed_result, nullptr, &err);
            if (err != CL_SUCCESS || !cache.device_buffer_result) {
                cache.device_buffer_result = nullptr;
                cache.capacity_result = 0;
                goto naive_fallback_cl;
            }
            cache.capacity_result = needed_result;
        }

        cl_mem d_a = cache.device_buffer_a;
        cl_mem d_b = cache.device_buffer_b;
        cl_mem d_result = cache.device_buffer_result;

        err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0, needed_a, local_a, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { goto naive_fallback_cl; }
        
        err = clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0, needed_b, local_b, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) { goto naive_fallback_cl; }
        
        if (beta != 0.0f) {
            err = clEnqueueWriteBuffer(queue, d_result, CL_TRUE, 0, needed_result, result, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) { goto naive_fallback_cl; }
        }

        int status = CLBlastSgemm(CLBlastLayoutRowMajor,
                     transpose_a ? CLBlastTransposeYes : CLBlastTransposeNo,
                     transpose_b ? CLBlastTransposeYes : CLBlastTransposeNo,
                     rows_a, cols_b, inner_dimension,
                     alpha,
                     d_a, 0, transpose_a ? rows_a : inner_dimension,
                     d_b, 0, transpose_b ? inner_dimension : cols_b,
                     beta,
                     d_result, 0, cols_b,
                     &queue, nullptr);

        if (status == 0) {
            clEnqueueReadBuffer(queue, d_result, CL_TRUE, 0, needed_result, result, 0, nullptr, nullptr);
            return;
        } else {
            static bool p = false; if (!p) { std::cout << "[CL] CLBlastSgemm Error: " << status << "\n"; p = true; }
        }
    }
naive_fallback_cl:
#endif
#ifdef QQ_BLAS_GENERIC
    cblas_sgemm(CblasRowMajor,
                transpose_a ? CblasTrans : CblasNoTrans,
                transpose_b ? CblasTrans : CblasNoTrans,
                rows_a, cols_b, inner_dimension,
                alpha,
                local_a, transpose_a ? rows_a : inner_dimension,
                local_b, transpose_b ? inner_dimension : cols_b,
                beta,
                result, cols_b);
    return;
#endif
    if (!transpose_a && !transpose_b) {
        if (beta == 0.0f) {
            for (int i = 0; i < rows_a * cols_b; ++i) {
                result[i] = 0.0f;
            }
        } else if (beta != 1.0f) {
            for (int i = 0; i < rows_a * cols_b; ++i) {
                result[i] *= beta;
            }
        }
        for (int i = 0; i < rows_a; ++i) {
            for (int k = 0; k < inner_dimension; ++k) {
                float term = alpha * local_a[i * inner_dimension + k];
                for (int j = 0; j < cols_b; ++j) {
                    result[i * cols_b + j] += term * local_b[k * cols_b + j];
                }
            }
        }
    } else if (!transpose_a && transpose_b) {
        for (int i = 0; i < rows_a; ++i) {
            for (int j = 0; j < cols_b; ++j) {
                float accumulator = 0.0f;
                for (int k = 0; k < inner_dimension; ++k) {
                    accumulator += local_a[i * inner_dimension + k] * local_b[j * inner_dimension + k];
                }
                result[i * cols_b + j] = beta * result[i * cols_b + j] + alpha * accumulator;
            }
        }
    } else if (transpose_a && !transpose_b) {
        if (beta == 0.0f) {
            for (int i = 0; i < rows_a * cols_b; ++i) {
                result[i] = 0.0f;
            }
        } else if (beta != 1.0f) {
            for (int i = 0; i < rows_a * cols_b; ++i) {
                result[i] *= beta;
            }
        }
        for (int k = 0; k < inner_dimension; ++k) {
            for (int i = 0; i < rows_a; ++i) {
                float term = alpha * local_a[k * rows_a + i];
                for (int j = 0; j < cols_b; ++j) {
                    result[i * cols_b + j] += term * local_b[k * cols_b + j];
                }
            }
        }
    } else {
        for (int i = 0; i < rows_a; ++i) {
            for (int j = 0; j < cols_b; ++j) {
                float accumulator = 0.0f;
                for (int k = 0; k < inner_dimension; ++k) {
                    accumulator += local_a[k * rows_a + i] * local_b[j * inner_dimension + k];
                }
                result[i * cols_b + j] = beta * result[i * cols_b + j] + alpha * accumulator;
            }
        }
    }
}

void matmult(const float* matrix_a, const float* matrix_b, float* result,
             int rows_a, int inner_dimension, int cols_b,
             bool transpose_a, bool transpose_b,
             float alpha, float beta,
             int split_into, MatmulInnerHook inner_hook,
             const float* next_weights, int next_weights_size) {
    preload();
    bool measure_tops = (std::rand() % 10000 == 0);
    auto start_time = measure_tops ? std::chrono::high_resolution_clock::now() : std::chrono::time_point<std::chrono::high_resolution_clock>();
    
    int final_split_into = split_into;
#ifdef QQ_BLAS_CLBLAST
    if (final_split_into <= 1 && !transpose_a) {
        double total_flops = 2.0 * rows_a * inner_dimension * cols_b;
        if (total_flops > 4.0e9) {
            double target_chunks = total_flops / 4.0e9;
            final_split_into = static_cast<int>(std::ceil(target_chunks));
            if (final_split_into > rows_a) {
                final_split_into = rows_a;
            }
        }
    }
#endif

    if (final_split_into > rows_a) {
        throw std::runtime_error("More chunks than rows in A");
    }
    
    if (final_split_into > 1 && transpose_a) {
        throw std::runtime_error("split_into > 1 not supported with transpose_a=true");
    }
    
    if (final_split_into <= 1) {
        matmult_impl(matrix_a, matrix_b, result, rows_a, inner_dimension, cols_b, transpose_a, transpose_b, alpha, beta, next_weights, next_weights_size);
        if (inner_hook) inner_hook(matrix_a, matrix_b, result, rows_a, inner_dimension, cols_b, 0, 1);
    } else {
        int chunk_rows = rows_a / final_split_into;
        for (int i = 0; i < final_split_into; ++i) {
            int start_row = i * chunk_rows;
            int current_rows = (i == final_split_into - 1) ? (rows_a - start_row) : chunk_rows;
            const float* current_a = matrix_a + start_row * inner_dimension;
            float* current_result = result + start_row * cols_b;
            const float* chunk_next_weights = (i == final_split_into - 1) ? next_weights : nullptr;
            int chunk_next_weights_size = (i == final_split_into - 1) ? next_weights_size : 0;
            matmult_impl(current_a, matrix_b, current_result, current_rows, inner_dimension, cols_b, transpose_a, transpose_b, alpha, beta, chunk_next_weights, chunk_next_weights_size);
            if (inner_hook) inner_hook(current_a, matrix_b, current_result, current_rows, inner_dimension, cols_b, i, final_split_into);
        }
    }
    
    if (measure_tops) {
        auto end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end_time - start_time;
        double ops = 2.0 * rows_a * cols_b * inner_dimension;
        double tops = (ops / duration.count()) / 1e12;
#ifdef QQ_BLAS_CLBLAST
        std::cout << "[TOPS GPU] " << rows_a << "x" << inner_dimension << "x" << cols_b << " : " << tops << " TOPS\n";
#else
        std::cout << "[TOPS CPU] " << rows_a << "x" << inner_dimension << "x" << cols_b << " : " << tops << " TOPS\n";
#endif
    }
}
#endif