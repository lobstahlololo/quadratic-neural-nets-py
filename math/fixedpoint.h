#ifndef QQ_FIXEDPOINT_H
#define QQ_FIXEDPOINT_H
#include <cstdint>
#include <cstring>
#include <mutex>
#include <cmath>
#include <limits>
#include <new>
inline uint32_t xorshift_32bits(uint32_t& state) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}
inline uint64_t xorshift_64bits(uint64_t& state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}
template<typename WordType, typename OutType>
struct StochasticRoundingKernel {
    static inline void process_block(const float* input, OutType* output, int start_idx, int count, WordType curr, int precision, uint32_t mask) {
        for (int j = 0; j < count; ++j) {
            int idx = start_idx + j;
            float value = input[idx];
            uint32_t u;
            std::memcpy(&u, &value, sizeof(uint32_t));
            uint32_t abs_u = u & 0x7FFFFFFF;
            uint32_t exponent = (abs_u >> 23) & 0xFF;
            uint32_t random_bits = (curr >> (j * precision)) & mask;
            uint32_t val = (1U << 23) | (abs_u & 0x7FFFFF);
            int shift = 150 - precision - exponent;
            uint32_t scaled_fixed = (shift >= 0 && shift < 32) ? (val >> shift) : ((shift < 0) ? (val << -shift) : 0);
            uint32_t last_n_sigfigs = scaled_fixed & mask;
            uint32_t integer_part = scaled_fixed >> precision;
            uint32_t rounded_magnitude = integer_part + ((last_n_sigfigs + random_bits) >> precision);
            int32_t sign_mask = (int32_t)u >> 31;
            int32_t rounded_int = (rounded_magnitude + sign_mask) ^ sign_mask;
            output[idx] = (exponent == 0) ? (OutType)0 : (OutType)rounded_int;
        }
    }
};
template<typename OutType>
inline void stochastic_round_array(const float* input, OutType* output, int size, int precision = 4) {
    uint64_t seed0 = 2463534242ULL;
    uint64_t seed1 = 4123546731ULL;
    uint64_t seed2 = 1290384712ULL;
    uint64_t seed3 = 9812403981ULL;
    int elements_per_word = 64 / precision;
    if (elements_per_word <= 0) {
        elements_per_word = 1;
    }
    int unroll_factor = 4;
    int elements_per_block = elements_per_word * unroll_factor;
    uint32_t mask = (1U << precision) - 1;
    int i = 0;
    for (; i <= size - elements_per_block; i += elements_per_block) {
        uint64_t curr0 = xorshift_64bits(seed0);
        uint64_t curr1 = xorshift_64bits(seed1);
        uint64_t curr2 = xorshift_64bits(seed2);
        uint64_t curr3 = xorshift_64bits(seed3);
        StochasticRoundingKernel<uint64_t, OutType>::process_block(input, output, i, elements_per_word, curr0, precision, mask);
        StochasticRoundingKernel<uint64_t, OutType>::process_block(input, output, i + elements_per_word, elements_per_word, curr1, precision, mask);
        StochasticRoundingKernel<uint64_t, OutType>::process_block(input, output, i + 2 * elements_per_word, elements_per_word, curr2, precision, mask);
        StochasticRoundingKernel<uint64_t, OutType>::process_block(input, output, i + 3 * elements_per_word, elements_per_word, curr3, precision, mask);
    }
    if (i < size) {
        uint64_t seed_rem = 2463534242ULL;
        uint64_t curr = xorshift_64bits(seed_rem);
        for (int j = 0; i < size; ++i, ++j) {
            float value = input[i];
            uint32_t u;
            std::memcpy(&u, &value, sizeof(uint32_t));
            uint32_t abs_u = u & 0x7FFFFFFF;
            uint32_t exponent = (abs_u >> 23) & 0xFF;
            uint32_t random_bits = (curr >> ((j % elements_per_word) * precision)) & mask;
            uint32_t val = (1U << 23) | (abs_u & 0x7FFFFF);
            int shift = 150 - precision - exponent;
            uint32_t scaled_fixed = (shift >= 0 && shift < 32) ? (val >> shift) : ((shift < 0) ? (val << -shift) : 0);
            uint32_t last_n_sigfigs = scaled_fixed & mask;
            uint32_t integer_part = scaled_fixed >> precision;
            uint32_t rounded_magnitude = integer_part + ((last_n_sigfigs + random_bits) >> precision);
            int32_t sign_mask = (int32_t)u >> 31;
            int32_t rounded_int = (rounded_magnitude + sign_mask) ^ sign_mask;
            output[i] = (exponent == 0) ? (OutType)0 : (OutType)rounded_int;
            if ((j + 1) % elements_per_word == 0 && i + 1 < size) {
                curr = xorshift_64bits(seed_rem);
            }
        }
    }
}
struct BlockHeader {
    size_t size;
    bool is_free;
    BlockHeader* next;
    BlockHeader* prev;
    uint8_t padding[128 - sizeof(size_t) - sizeof(bool) - 2 * sizeof(BlockHeader*)];
};
class Scratchpad {
private:
    inline static uint8_t* buffer = nullptr;
    inline static size_t total_size = 32 * 1024 * 1024;
    inline static BlockHeader* head = nullptr;
    inline static std::mutex mtx;
    static void initialize_if_needed() {
        if (!buffer) {
            buffer = (uint8_t*)::operator new[](total_size, std::align_val_t{128});
            head = (BlockHeader*)buffer;
            head->size = total_size - sizeof(BlockHeader);
            head->is_free = true;
            head->next = nullptr;
            head->prev = nullptr;
        }
    }
public:
    static void set_size(size_t new_size) {
        std::lock_guard<std::mutex> lock(mtx);
        if (buffer) {
            ::operator delete[](buffer, std::align_val_t{128});
            buffer = nullptr;
            head = nullptr;
        }
        total_size = new_size;
    }
    static void* alloc(size_t size) {
        std::lock_guard<std::mutex> lock(mtx);
        initialize_if_needed();
        size = (size + 127) & ~127;
        BlockHeader* curr = head;
        while (curr) {
            if (curr->is_free && curr->size >= size) {
                if (curr->size >= size + sizeof(BlockHeader) + 128) {
                    BlockHeader* next_block = (BlockHeader*)((uint8_t*)curr + sizeof(BlockHeader) + size);
                    next_block->size = curr->size - size - sizeof(BlockHeader);
                    next_block->is_free = true;
                    next_block->next = curr->next;
                    next_block->prev = curr;
                    if (curr->next) curr->next->prev = next_block;
                    curr->next = next_block;
                    curr->size = size;
                }
                curr->is_free = false;
                return (void*)((uint8_t*)curr + sizeof(BlockHeader));
            }
            curr = curr->next;
        }
        return ::operator new[](size, std::align_val_t{128});
    }
    static void free(void* ptr) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mtx);
        if (buffer && (uint8_t*)ptr >= buffer && (uint8_t*)ptr < buffer + total_size) {
            BlockHeader* curr = (BlockHeader*)((uint8_t*)ptr - sizeof(BlockHeader));
            curr->is_free = true;
            if (curr->next && curr->next->is_free) {
                curr->size += sizeof(BlockHeader) + curr->next->size;
                curr->next = curr->next->next;
                if (curr->next) curr->next->prev = curr;
            }
            if (curr->prev && curr->prev->is_free) {
                curr->prev->size += sizeof(BlockHeader) + curr->size;
                curr->prev->next = curr->next;
                if (curr->next) curr->next->prev = curr->prev;
            }
        } else {
            ::operator delete[](ptr, std::align_val_t{128});
        }
    }
};
struct ScratchpadBuffer {
    float* ptr;
    ScratchpadBuffer(size_t size) {
        ptr = (float*)Scratchpad::alloc(size * sizeof(float));
    }
    ~ScratchpadBuffer() {
        Scratchpad::free(ptr);
    }
};
template<typename T>
struct FixedPointBlock {
    T* mantissa;
    int32_t* exponents;
    int rows;
    int cols;
    int block_size;
    bool block_columns;
    int num_blocks;

    FixedPointBlock(int r, int c, int bs, bool bc) {
        rows = r;
        cols = c;
        block_size = bs;
        block_columns = bc;
        mantissa = (T*)Scratchpad::alloc(rows * cols * sizeof(T));
        if (block_columns) {
            num_blocks = (cols + block_size - 1) / block_size;
            exponents = (int32_t*)Scratchpad::alloc(rows * num_blocks * sizeof(int32_t));
        } else {
            num_blocks = (rows + block_size - 1) / block_size;
            exponents = (int32_t*)Scratchpad::alloc(num_blocks * cols * sizeof(int32_t));
        }
    }

    ~FixedPointBlock() {
        Scratchpad::free(mantissa);
        Scratchpad::free(exponents);
    }

    void fit_exponent(const float* values) {
        if (block_columns) {
            int num_b = (cols + block_size - 1) / block_size;
            for (int i = 0; i < rows; ++i) {
                for (int b = 0; b < num_b; ++b) {
                    uint32_t max_biased = 0;
                    int start_col = b * block_size;
                    int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                    for (int j = start_col; j < end_col; ++j) {
                        uint32_t abs_val;
                        std::memcpy(&abs_val, &values[i * cols + j], sizeof(uint32_t));
                        abs_val &= 0x7FFFFFFF;
                        uint32_t biased = (abs_val >> 23) & 0xFF;
                        if (biased > max_biased) {
                            max_biased = biased;
                        }
                    }
                    exponents[i * num_b + b] = (int32_t)max_biased - 127;
                }
            }
        } else {
            int num_b = (rows + block_size - 1) / block_size;
            for (int j = 0; j < cols; ++j) {
                for (int b = 0; b < num_b; ++b) {
                    uint32_t max_biased = 0;
                    int start_row = b * block_size;
                    int end_row = (start_row + block_size < rows) ? (start_row + block_size) : rows;
                    for (int i = start_row; i < end_row; ++i) {
                        uint32_t abs_val;
                        std::memcpy(&abs_val, &values[i * cols + j], sizeof(uint32_t));
                        abs_val &= 0x7FFFFFFF;
                        uint32_t biased = (abs_val >> 23) & 0xFF;
                        if (biased > max_biased) {
                            max_biased = biased;
                        }
                    }
                    exponents[b * cols + j] = (int32_t)max_biased - 127;
                }
            }
        }
    }

    void floats_to_mantissa(const float* floats, int precision = 4) {
        if (block_columns) {
            float* temp = new float[cols];
            int num_b = (cols + block_size - 1) / block_size;
            for (int i = 0; i < rows; ++i) {
                for (int b = 0; b < num_b; ++b) {
                    int32_t exp = exponents[i * num_b + b];
                    float scale = std::ldexp(1.0f, -exp);
                    int start_col = b * block_size;
                    int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                    int count = end_col - start_col;
                    for (int j = 0; j < count; ++j) {
                        temp[j] = floats[i * cols + start_col + j] * scale;
                    }
                    stochastic_round_array(temp, &mantissa[i * cols + start_col], count, precision);
                }
            }
            delete[] temp;
        } else {
            float* temp = new float[rows];
            int num_b = (rows + block_size - 1) / block_size;
            for (int j = 0; j < cols; ++j) {
                for (int b = 0; b < num_b; ++b) {
                    int32_t exp = exponents[b * cols + j];
                    float scale = std::ldexp(1.0f, -exp);
                    int start_row = b * block_size;
                    int end_row = (start_row + block_size < rows) ? (start_row + block_size) : rows;
                    int count = end_row - start_row;
                    for (int i = 0; i < count; ++i) {
                        temp[i] = floats[(start_row + i) * cols + j] * scale;
                    }
                    T* col_mant = new T[count];
                    stochastic_round_array(temp, col_mant, count, precision);
                    for (int i = 0; i < count; ++i) {
                        mantissa[(start_row + i) * cols + j] = col_mant[i];
                    }
                    delete[] col_mant;
                }
            }
            delete[] temp;
        }
    }

    void mantissa_to_floats(float* output, int precision = 4) {
        if (block_columns) {
            int num_b = (cols + block_size - 1) / block_size;
            for (int i = 0; i < rows; ++i) {
                for (int b = 0; b < num_b; ++b) {
                    int32_t exp = exponents[i * num_b + b];
                    float scale = std::ldexp(1.0f, exp - precision);
                    int start_col = b * block_size;
                    int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                    for (int j = start_col; j < end_col; ++j) {
                        output[i * cols + j] = (float)mantissa[i * cols + j] * scale;
                    }
                }
            }
        } else {
            int num_b = (rows + block_size - 1) / block_size;
            for (int j = 0; j < cols; ++j) {
                for (int b = 0; b < num_b; ++b) {
                    int32_t exp = exponents[b * cols + j];
                    float scale = std::ldexp(1.0f, exp - precision);
                    int start_row = b * block_size;
                    int end_row = (start_row + block_size < rows) ? (start_row + block_size) : rows;
                    for (int i = start_row; i < end_row; ++i) {
                        output[i * cols + j] = (float)mantissa[i * cols + j] * scale;
                    }
                }
            }
        }
    }

    void mantissa_to_floats_product(float* output, int precision = 4) {
        int num_b = (cols + block_size - 1) / block_size;
        for (int i = 0; i < rows; ++i) {
            for (int b = 0; b < num_b; ++b) {
                int32_t exp = exponents[i * num_b + b];
                float scale = std::ldexp(1.0f, exp - 2 * precision);
                int start_col = b * block_size;
                int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                for (int j = start_col; j < end_col; ++j) {
                    output[i * cols + j] = (float)mantissa[i * cols + j] * scale;
                }
            }
        }
    }
};
#endif