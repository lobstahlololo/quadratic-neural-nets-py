#include <iostream>
#include <vector>
#include <chrono>
#include "math.h"
int main() {
    int rows_a = 2048*4;
    int inner_dimension = 2048*4;
    int cols_b = 2048*4;
    int iterations = 10;
    std::vector<float> matrix_a(rows_a * inner_dimension, 1.0f);
    std::vector<float> matrix_b(inner_dimension * cols_b, 1.0f);
    std::vector<float> result(rows_a * cols_b, 0.0f);
    for (int i = 0; i < 2; ++i) {
        matmult(matrix_a.data(), matrix_b.data(), result.data(), rows_a, inner_dimension, cols_b, false, false, 1.0f, 0.0f);
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        matmult(matrix_a.data(), matrix_b.data(), result.data(), rows_a, inner_dimension, cols_b, false, false, 1.0f, 0.0f);
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> total_duration = end_time - start_time;
    double average_duration = total_duration.count() / iterations;
    double operations_per_matmul = 2.0 * static_cast<double>(rows_a) * static_cast<double>(cols_b) * static_cast<double>(inner_dimension);
    double tops = (operations_per_matmul / average_duration) / 1e12;
    std::cout << "Matrix Dimensions: " << rows_a << "x" << inner_dimension << "x" << cols_b << "\n";
    std::cout << "Average Time: " << average_duration * 1000.0 << " ms\n";
    std::cout << "Estimated Performance: " << tops << " TOPS\n";
    return 0;
}
