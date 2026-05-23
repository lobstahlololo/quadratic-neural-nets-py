#ifndef QQ_NPUBLAS_LOADED
#define QQ_NPUBLAS_LOADED
#include <cstdint>
#include <dlfcn.h>
namespace npublas {
inline void* stub_library = nullptr;
inline void* rpc_library = nullptr;
inline uint64_t session_handle = 0;
inline bool load_attempted = false;
inline bool tier1_ok = false;
inline bool tier2_ok = false;
typedef int (*open_fn_t)(const char*, uint64_t*);
typedef int (*matmul_8bit_fn_t)(uint64_t, const float*, int, const float*, int, float*, int, int, int, int);
inline open_fn_t stub_open_fn = nullptr;
inline matmul_8bit_fn_t stub_matmul_fn = nullptr;
typedef int (*rpc_open_fn_t)(const char*, uint64_t*);
typedef int (*rpc_invoke_fn_t)(uint64_t, uint32_t, const void*, int, void*, int);
inline rpc_open_fn_t rpc_open_fn = nullptr;
inline rpc_invoke_fn_t rpc_invoke_fn = nullptr;
inline void try_stub() {
    stub_library = dlopen("libnputils.so", RTLD_NOW);
    if (!stub_library) return;
    stub_open_fn = (open_fn_t)dlsym(stub_library, "nputils_open");
    stub_matmul_fn = (matmul_8bit_fn_t)dlsym(stub_library, "nputils_matmul_8bit");
    if (!stub_open_fn || !stub_matmul_fn) {
        dlclose(stub_library);
        stub_library = nullptr;
        stub_open_fn = nullptr;
        stub_matmul_fn = nullptr;
        return;
    }
    int err = stub_open_fn("nputils", &session_handle);
    if (err != 0) {
        dlclose(stub_library);
        stub_library = nullptr;
        stub_open_fn = nullptr;
        stub_matmul_fn = nullptr;
        return;
    }
    tier1_ok = true;
}
inline void try_direct_rpc() {
    rpc_library = dlopen("libadsprpc.so", RTLD_NOW);
    if (!rpc_library) return;
    rpc_open_fn = (rpc_open_fn_t)dlsym(rpc_library, "remote_handle64_open");
    rpc_invoke_fn = (rpc_invoke_fn_t)dlsym(rpc_library, "remote_handle64_invoke");
    if (!rpc_open_fn || !rpc_invoke_fn) {
        dlclose(rpc_library);
        rpc_library = nullptr;
        rpc_open_fn = nullptr;
        rpc_invoke_fn = nullptr;
        return;
    }
    int err = rpc_open_fn("nputils", &session_handle);
    if (err != 0) {
        dlclose(rpc_library);
        rpc_library = nullptr;
        rpc_open_fn = nullptr;
        rpc_invoke_fn = nullptr;
        return;
    }
    tier2_ok = true;
}
inline void ensure_initialized() {
    if (load_attempted) return;
    load_attempted = true;
    try_stub();
    if (tier1_ok) return;
    try_direct_rpc();
}
inline bool npu_available() {
    ensure_initialized();
    return tier1_ok || tier2_ok;
}
inline int matmul_8bit(const float* matrix_a, const float* matrix_b, float* result,
                       int rows_a, int inner_dimension, int cols_b) {
    if (tier1_ok) {
        int input_len = inner_dimension * cols_b;
        int weights_len = rows_a * inner_dimension;
        int output_len = rows_a * cols_b;
        return stub_matmul_fn(session_handle,
                              matrix_a, input_len,
                              matrix_b, weights_len,
                              result, output_len,
                              rows_a, inner_dimension, cols_b);
    }
    if (tier2_ok) {
        int input_len = inner_dimension * cols_b;
        int weights_len = rows_a * inner_dimension;
        int output_len = rows_a * cols_b;
        int total_in_bytes = 2 * sizeof(int) + input_len * sizeof(float) + weights_len * sizeof(float);
        int total_out_bytes = sizeof(int) + output_len * sizeof(float);
        unsigned char* inbuf = new unsigned char[total_in_bytes];
        unsigned char* outbuf = new unsigned char[total_out_bytes];
        int offset = 0;
        *(int*)(inbuf + offset) = input_len; offset += sizeof(int);
        memcpy(inbuf + offset, matrix_a, input_len * sizeof(float)); offset += input_len * sizeof(float);
        *(int*)(inbuf + offset) = weights_len; offset += sizeof(int);
        memcpy(inbuf + offset, matrix_b, weights_len * sizeof(float)); offset += weights_len * sizeof(float);
        *(int*)(inbuf + offset) = output_len; offset += sizeof(int);
        *(int*)(inbuf + offset) = rows_a; offset += sizeof(int);
        *(int*)(inbuf + offset) = inner_dimension; offset += sizeof(int);
        *(int*)(inbuf + offset) = cols_b; offset += sizeof(int);
        int method_id = 0;
        int err = rpc_invoke_fn(session_handle, method_id, inbuf, total_in_bytes, outbuf, total_out_bytes);
        if (err == 0) {
            memcpy(result, outbuf + sizeof(int), output_len * sizeof(float));
        }
        delete[] inbuf;
        delete[] outbuf;
        return err;
    }
    return -1;
}
}
#endif