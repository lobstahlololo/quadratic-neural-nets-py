#ifndef QQ_NPUBLAS_LOADED
#define QQ_NPUBLAS_LOADED
#include <cstdint>
#include <dlfcn.h>
namespace npublas {
inline void* library_handle = nullptr;
inline uint64_t session_handle = 0;
inline bool initialized = false;
inline bool load_attempted = false;
typedef int (*open_fn_t)(const char*, uint64_t*);
typedef int (*matmul_8bit_fn_t)(uint64_t, const float*, int, const float*, int, float*, int, int, int, int);
inline open_fn_t open_fn = nullptr;
inline matmul_8bit_fn_t matmul_8bit_fn = nullptr;
inline void ensure_initialized() {
    if (load_attempted) return;
    load_attempted = true;
    library_handle = dlopen("libnputils.so", RTLD_NOW);
    if (!library_handle) return;
    open_fn = (open_fn_t)dlsym(library_handle, "nputils_open");
    matmul_8bit_fn = (matmul_8bit_fn_t)dlsym(library_handle, "nputils_matmul_8bit");
    if (!open_fn || !matmul_8bit_fn) {
        dlclose(library_handle);
        library_handle = nullptr;
        open_fn = nullptr;
        matmul_8bit_fn = nullptr;
        return;
    }
    int err = open_fn("nputils", &session_handle);
    if (err != 0) {
        dlclose(library_handle);
        library_handle = nullptr;
        open_fn = nullptr;
        matmul_8bit_fn = nullptr;
        return;
    }
    initialized = true;
}
inline bool npu_available() {
    ensure_initialized();
    return initialized;
}
inline int matmul_8bit(const float* matrix_a, const float* matrix_b, float* result,
                       int rows_a, int inner_dimension, int cols_b) {
    int input_len = inner_dimension * cols_b;
    int weights_len = rows_a * inner_dimension;
    int output_len = rows_a * cols_b;
    return matmul_8bit_fn(session_handle,
                          matrix_a, input_len,
                          matrix_b, weights_len,
                          result, output_len,
                          rows_a, inner_dimension, cols_b);
}
}
#endif