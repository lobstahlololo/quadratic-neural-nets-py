#ifndef TFLITE_MANAGER_H
#define TFLITE_MANAGER_H
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/kernels/register.h"


#include "tensorflow/lite/delegates/gpu/delegate.h"               
#include "tensorflow/lite/delegates/external/external_delegate.h"  
 
/*
std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("/data/data/com.termux/files/home/cache/matmul.tflite");



*/

#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include "../fixedpoint.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/delegates/external/external_delegate.h"
#include "tensorflow/lite/delegates/gpu/delegate.h"

struct Model {
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    int M, K, N;
};

inline std::vector<Model> models;

inline Model create_model(int M, int K, int N) {
    std::string filename = "matmul_int8_" + std::to_string(M) + "x" + std::to_string(K) + "_" + std::to_string(K) + "x" + std::to_string(N) + ".tflite";
    auto model = tflite::FlatBufferModel::BuildFromFile(filename.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<tflite::Interpreter> interpreter;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    return Model{std::move(model), std::move(interpreter), M, K, N};
}

inline Model& fetch_model(int M, int K, int N) {
    for (size_t i = 0; i < models.size(); ++i) {
        if (models[i].M == M && models[i].K == K && models[i].N == N) {
            return models[i];
        }
    }
    models.push_back(create_model(M, K, N));
    return models.back();
}

inline void matmul_int8(const float* A, const float* B, float* C, int M, int K, int N, bool transposeA, bool transposeB, bool use_dsp = true, bool fix_left = true) {
    int target_precision = 8;
    FixedPointBlock<int8_t> fixed_A(M, K, K, true);
    FixedPointBlock<int8_t> fixed_B(K, N, K, false);
    if (fix_left) {
        fixed_A.fit_exponent(A);
        fixed_A.floats_to_mantissa(A, 0, target_precision);
        fixed_B.fit_exponent(B);
        fixed_B.floats_to_mantissa(B, 0, 0, 3, target_precision, fixed_A.std_dev);
    } else {
        fixed_B.fit_exponent(B);
        fixed_B.floats_to_mantissa(B, 0, target_precision);
        fixed_A.fit_exponent(A);
        fixed_A.floats_to_mantissa(A, 0, 0, 3, target_precision, fixed_B.std_dev);
    }
    Model& model_raw = fetch_model(M, K, N);
    tflite::Interpreter* interpreter = model_raw.interpreter.get();
    if (!interpreter) return;
    if (use_dsp) {
        TfLiteExternalDelegateOptions options = TfLiteExternalDelegateOptionsDefault("libQnnTFLiteDelegate.so");
        options.insert(&options, "backend_type", "DSP");
        TfLiteDelegate* dsp_delegate = TfLiteExternalDelegateCreate(&options);
        if (dsp_delegate) {
            interpreter->ModifyGraphWithDelegate(dsp_delegate);
        }
    } else {
        auto gpu_opts = TfLiteGpuDelegateOptionsV2Default();
        gpu_opts.precision_loss_allowed = 1;
        gpu_opts.inference_priority1 = TFLITE_GPU_INFERENCE_PRIORITY_MIN_LATENCY;
        TfLiteDelegate* gpu_delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
        if (gpu_delegate) {
            interpreter->ModifyGraphWithDelegate(gpu_delegate);
        }
    }
    if (interpreter->AllocateTensors() != kTfLiteOk) return;
    int8_t* input_a = interpreter->typed_input_tensor<int8_t>(0);
    int8_t* input_b = interpreter->typed_input_tensor<int8_t>(1);
    if (input_a && input_b) {
        std::memcpy(input_a, fixed_A.mantissa, M * K * sizeof(int8_t));
        std::memcpy(input_b, fixed_B.mantissa, K * N * sizeof(int8_t));
    }
    if (interpreter->Invoke() != kTfLiteOk) return;
    int8_t* output = interpreter->typed_output_tensor<int8_t>(0);
    if (output) {
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                int block_flat_idx_a = i;
                int block_flat_idx_b = j;
                int32_t total_exp = fixed_A.exponents[block_flat_idx_a] + fixed_B.exponents[block_flat_idx_b];
                int32_t total_prec = fixed_A.precisions[block_flat_idx_a] + fixed_B.precisions[block_flat_idx_b];
                C[i * N + j] = (float)output[i * N + j] * std::ldexp(1.0f, total_exp - total_prec);
            }
        }
    }
}
#endif