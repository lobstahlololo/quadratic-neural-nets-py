#include "math/math.h"
#include <iostream>
#include <vector>
#include <cmath>
int main() {
    int m = 128;
    int k = 256;
    int n = 64;
    std::vector<float> a(m * k, 1.0f);
    std::vector<float> b(k * n, 2.0f);
    std::vector<float> c(m * n, 0.0f);
    std::cout << "Running matmult..." << std::endl;
    matmult(a.data(), b.data(), c.data(), m, k, n, false, false, 1.0f, 0.0f);
    float expected = k * 1.0f * 2.0f;
    std::cout << "Result [0]: " << c[0] << " (Expected: " << expected << ")" << std::endl;
    bool ok = true;
    for (int i = 0; i < m * n; ++i) {
        if (std::abs(c[i] - expected) > 1e-4) {
            ok = false;
            break;
        }
    }
    if (ok) {
        std::cout << "Test passed!" << std::endl;
    } else {
        std::cout << "Test failed!" << std::endl;
    }
    return 0;
}