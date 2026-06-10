#ifndef QQ_FIXEDPOINT_H
#define QQ_FIXEDPOINT_H
#include <cstdint>
#include <cstring>
#include <mutex>
#include <cmath>
#include <limits>
#include <new>
inline uint32_t xorshift_32bits(uint32_t& random_state) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}
inline uint64_t xorshift_64bits(uint64_t& random_state) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 7;
    random_state ^= random_state << 17;
    return random_state;
}
template<typename WordType, typename OutType>
struct StochasticRoundingKernel {
    static inline void process_block(const float* input_array, OutType* output_array, int start_index, int count, WordType current_random_word, int precision, uint32_t fractional_mask) {
        for (int element_index = 0; element_index < count; ++element_index) {
            int flat_index = start_index + element_index;
            float input_value = input_array[flat_index];
            uint32_t raw_bits;
            std::memcpy(&raw_bits, &input_value, sizeof(uint32_t));
            uint32_t absolute_raw_bits = raw_bits & 0x7FFFFFFF;
            uint32_t exponent_bits = (absolute_raw_bits >> 23) & 0xFF;
            uint32_t random_bits = (current_random_word >> (element_index * precision)) & fractional_mask;
            uint32_t mantissa_value = (1U << 23) | (absolute_raw_bits & 0x7FFFFF);
            int bit_shift = 150 - precision - exponent_bits;
            uint32_t scaled_fixed_value = (bit_shift >= 0 && bit_shift < 32) ? (mantissa_value >> bit_shift) : ((bit_shift < 0) ? (mantissa_value << -bit_shift) : 0);
            uint32_t fractional_bits_mask = scaled_fixed_value & fractional_mask;
            uint32_t integer_part_value = scaled_fixed_value >> precision;
            uint32_t rounded_magnitude = integer_part_value + ((fractional_bits_mask + random_bits) >> precision);
            int32_t sign_bit_mask = (int32_t)raw_bits >> 31;
            int32_t rounded_integer = (rounded_magnitude + sign_bit_mask) ^ sign_bit_mask;
            output_array[flat_index] = (exponent_bits == 0) ? (OutType)0 : (OutType)rounded_integer;
        }
    }
};
template<typename OutType>
inline void stochastic_round_array(const float* input_array, OutType* output_array, int array_size, int precision = 4) {
    uint64_t seed_word0 = 2463534242ULL;
    uint64_t seed_word1 = 4123546731ULL;
    uint64_t seed_word2 = 1290384712ULL;
    uint64_t seed_word3 = 9812403981ULL;
    int elements_per_word = 64 / precision;
    if (elements_per_word <= 0) {
        elements_per_word = 1;
    }
    int unroll_factor = 4;
    int elements_per_block = elements_per_word * unroll_factor;
    uint32_t fractional_mask = (1U << precision) - 1;
    int loop_index = 0;
    for (; loop_index <= array_size - elements_per_block; loop_index += elements_per_block) {
        uint64_t random_word0 = xorshift_64bits(seed_word0);
        uint64_t random_word1 = xorshift_64bits(seed_word1);
        uint64_t random_word2 = xorshift_64bits(seed_word2);
        uint64_t random_word3 = xorshift_64bits(seed_word3);
        StochasticRoundingKernel<uint64_t, OutType>::process_block(input_array, output_array, loop_index, elements_per_word, random_word0, precision, fractional_mask);
        StochasticRoundingKernel<uint64_t, OutType>::process_block(input_array, output_array, loop_index + elements_per_word, elements_per_word, random_word1, precision, fractional_mask);
        StochasticRoundingKernel<uint64_t, OutType>::process_block(input_array, output_array, loop_index + 2 * elements_per_word, elements_per_word, random_word2, precision, fractional_mask);
        StochasticRoundingKernel<uint64_t, OutType>::process_block(input_array, output_array, loop_index + 3 * elements_per_word, elements_per_word, random_word3, precision, fractional_mask);
    }
    if (loop_index < array_size) {
        uint64_t seed_word_remainder = 2463534242ULL;
        uint64_t current_random_word = xorshift_64bits(seed_word_remainder);
        for (int element_index = 0; loop_index < array_size; ++loop_index, ++element_index) {
            float input_value = input_array[loop_index];
            uint32_t raw_bits;
            std::memcpy(&raw_bits, &input_value, sizeof(uint32_t));
            uint32_t absolute_raw_bits = raw_bits & 0x7FFFFFFF;
            uint32_t exponent_bits = (absolute_raw_bits >> 23) & 0xFF;
            uint32_t random_bits = (current_random_word >> ((element_index % elements_per_word) * precision)) & fractional_mask;
            uint32_t mantissa_value = (1U << 23) | (absolute_raw_bits & 0x7FFFFF);
            int bit_shift = 150 - precision - exponent_bits;
            uint32_t scaled_fixed_value = (bit_shift >= 0 && bit_shift < 32) ? (mantissa_value >> bit_shift) : ((bit_shift < 0) ? (mantissa_value << -bit_shift) : 0);
            uint32_t fractional_bits_mask = scaled_fixed_value & fractional_mask;
            uint32_t integer_part_value = scaled_fixed_value >> precision;
            uint32_t rounded_magnitude = integer_part_value + ((fractional_bits_mask + random_bits) >> precision);
            int32_t sign_bit_mask = (int32_t)raw_bits >> 31;
            int32_t rounded_integer = (rounded_magnitude + sign_bit_mask) ^ sign_bit_mask;
            output_array[loop_index] = (exponent_bits == 0) ? (OutType)0 : (OutType)rounded_integer;
            if ((element_index + 1) % elements_per_word == 0 && loop_index + 1 < array_size) {
                current_random_word = xorshift_64bits(seed_word_remainder);
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
    inline static uint8_t* memory_buffer = nullptr;
    inline static size_t total_size = 32 * 1024 * 1024;
    inline static BlockHeader* head_block = nullptr;
    inline static std::mutex scratchpad_mutex;
    static void initialize_if_needed() {
        if (!memory_buffer) {
            memory_buffer = (uint8_t*)::operator new[](total_size, std::align_val_t{128});
            head_block = (BlockHeader*)memory_buffer;
            head_block->size = total_size - sizeof(BlockHeader);
            head_block->is_free = true;
            head_block->next = nullptr;
            head_block->prev = nullptr;
        }
    }
public:
    static void set_size(size_t new_size) {
        std::lock_guard<std::mutex> lock(scratchpad_mutex);
        if (memory_buffer) {
            ::operator delete[](memory_buffer, std::align_val_t{128});
            memory_buffer = nullptr;
            head_block = nullptr;
        }
        total_size = new_size;
    }
    static void* alloc(size_t size) {
        std::lock_guard<std::mutex> lock(scratchpad_mutex);
        initialize_if_needed();
        size = (size + 127) & ~127;
        BlockHeader* current_block = head_block;
        while (current_block) {
            if (current_block->is_free && current_block->size >= size) {
                if (current_block->size >= size + sizeof(BlockHeader) + 128) {
                    BlockHeader* next_block = (BlockHeader*)((uint8_t*)current_block + sizeof(BlockHeader) + size);
                    next_block->size = current_block->size - size - sizeof(BlockHeader);
                    next_block->is_free = true;
                    next_block->next = current_block->next;
                    next_block->prev = current_block;
                    if (current_block->next) current_block->next->prev = next_block;
                    current_block->next = next_block;
                    current_block->size = size;
                }
                current_block->is_free = false;
                return (void*)((uint8_t*)current_block + sizeof(BlockHeader));
            }
            current_block = current_block->next;
        }
        return ::operator new[](size, std::align_val_t{128});
    }
    static void free(void* allocated_pointer) {
        if (!allocated_pointer) return;
        std::lock_guard<std::mutex> lock(scratchpad_mutex);
        if (memory_buffer && (uint8_t*)allocated_pointer >= memory_buffer && (uint8_t*)allocated_pointer < memory_buffer + total_size) {
            BlockHeader* current_block = (BlockHeader*)((uint8_t*)allocated_pointer - sizeof(BlockHeader));
            current_block->is_free = true;
            if (current_block->next && current_block->next->is_free) {
                current_block->size += sizeof(BlockHeader) + current_block->next->size;
                current_block->next = current_block->next->next;
                if (current_block->next) current_block->next->prev = current_block;
            }
            if (current_block->prev && current_block->prev->is_free) {
                current_block->prev->size += sizeof(BlockHeader) + current_block->size;
                current_block->prev->next = current_block->next;
                if (current_block->next) current_block->next->prev = current_block->prev;
            }
        } else {
            ::operator delete[](allocated_pointer, std::align_val_t{128});
        }
    }
};
struct ScratchpadBuffer {
    float* pointer;
    ScratchpadBuffer(size_t size) {
        pointer = (float*)Scratchpad::alloc(size * sizeof(float));
    }
    ~ScratchpadBuffer() {
        Scratchpad::free(pointer);
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
    FixedPointBlock(int num_rows, int num_columns, int target_block_size, bool use_block_columns) {
        rows = num_rows;
        cols = num_columns;
        block_size = target_block_size;
        block_columns = use_block_columns;
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
            int num_blocks_count = (cols + block_size - 1) / block_size;
            for (int row_index = 0; row_index < rows; ++row_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    uint32_t max_biased = 0;
                    int start_col = block_index * block_size;
                    int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                    for (int col_index = start_col; col_index < end_col; ++col_index) {
                        uint32_t absolute_raw_bits;
                        std::memcpy(&absolute_raw_bits, &values[row_index * cols + col_index], sizeof(uint32_t));
                        absolute_raw_bits &= 0x7FFFFFFF;
                        uint32_t biased = (absolute_raw_bits >> 23) & 0xFF;
                        if (biased > max_biased) {
                            max_biased = biased;
                        }
                    }
                    exponents[row_index * num_blocks_count + block_index] = (int32_t)max_biased - 127;
                }
            }
        } else {
            int num_blocks_count = (rows + block_size - 1) / block_size;
            for (int col_index = 0; col_index < cols; ++col_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    uint32_t max_biased = 0;
                    int start_row = block_index * block_size;
                    int end_row = (start_row + block_size < rows) ? (start_row + block_size) : rows;
                    for (int row_index = start_row; row_index < end_row; ++row_index) {
                        uint32_t absolute_raw_bits;
                        std::memcpy(&absolute_raw_bits, &values[row_index * cols + col_index], sizeof(uint32_t));
                        absolute_raw_bits &= 0x7FFFFFFF;
                        uint32_t biased = (absolute_raw_bits >> 23) & 0xFF;
                        if (biased > max_biased) {
                            max_biased = biased;
                        }
                    }
                    exponents[block_index * cols + col_index] = (int32_t)max_biased - 127;
                }
            }
        }
    }
    void floats_to_mantissa(const float* floats_array, int precision = 4) {
        if (block_columns) {
            float* temporary_buffer = new float[cols];
            int num_blocks_count = (cols + block_size - 1) / block_size;
            for (int row_index = 0; row_index < rows; ++row_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    int32_t exponent_val = exponents[row_index * num_blocks_count + block_index];
                    float exponent_scale = std::ldexp(1.0f, -exponent_val);
                    int start_col = block_index * block_size;
                    int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                    int count = end_col - start_col;
                    for (int col_index = 0; col_index < count; ++col_index) {
                        temporary_buffer[col_index] = floats_array[row_index * cols + start_col + col_index] * exponent_scale;
                    }
                    stochastic_round_array(temporary_buffer, &mantissa[row_index * cols + start_col], count, precision);
                }
            }
            delete[] temporary_buffer;
        } else {
            float* temporary_buffer = new float[rows];
            int num_blocks_count = (rows + block_size - 1) / block_size;
            for (int col_index = 0; col_index < cols; ++col_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    int32_t exponent_val = exponents[block_index * cols + col_index];
                    float exponent_scale = std::ldexp(1.0f, -exponent_val);
                    int start_row = block_index * block_size;
                    int end_row = (start_row + block_size < rows) ? (start_row + block_size) : rows;
                    int count = end_row - start_row;
                    for (int row_index = 0; row_index < count; ++row_index) {
                        temporary_buffer[row_index] = floats_array[(start_row + row_index) * cols + col_index] * exponent_scale;
                    }
                    T* column_mantissa = new T[count];
                    stochastic_round_array(temporary_buffer, column_mantissa, count, precision);
                    for (int row_index = 0; row_index < count; ++row_index) {
                        mantissa[(start_row + row_index) * cols + col_index] = column_mantissa[row_index];
                    }
                    delete[] column_mantissa;
                }
            }
            delete[] temporary_buffer;
        }
    }
    void mantissa_to_floats(float* output_array, int precision = 4) {
        if (block_columns) {
            int num_blocks_count = (cols + block_size - 1) / block_size;
            for (int row_index = 0; row_index < rows; ++row_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    int32_t exponent_val = exponents[row_index * num_blocks_count + block_index];
                    float exponent_scale = std::ldexp(1.0f, exponent_val - precision);
                    int start_col = block_index * block_size;
                    int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                    for (int col_index = start_col; col_index < end_col; ++col_index) {
                        output_array[row_index * cols + col_index] = (float)mantissa[row_index * cols + col_index] * exponent_scale;
                    }
                }
            }
        } else {
            int num_blocks_count = (rows + block_size - 1) / block_size;
            for (int col_index = 0; col_index < cols; ++col_index) {
                for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                    int32_t exponent_val = exponents[block_index * cols + col_index];
                    float exponent_scale = std::ldexp(1.0f, exponent_val - precision);
                    int start_row = block_index * block_size;
                    int end_row = (start_row + block_size < rows) ? (start_row + block_size) : rows;
                    for (int row_index = start_row; row_index < end_row; ++row_index) {
                        output_array[row_index * cols + col_index] = (float)mantissa[row_index * cols + col_index] * exponent_scale;
                    }
                }
            }
        }
    }
    void mantissa_to_floats_product(float* output_array, int precision = 4) {
        int num_blocks_count = (cols + block_size - 1) / block_size;
        for (int row_index = 0; row_index < rows; ++row_index) {
            for (int block_index = 0; block_index < num_blocks_count; ++block_index) {
                int32_t exponent_val = exponents[row_index * num_blocks_count + block_index];
                float exponent_scale = std::ldexp(1.0f, exponent_val - 2 * precision);
                int start_col = block_index * block_size;
                int end_col = (start_col + block_size < cols) ? (start_col + block_size) : cols;
                for (int col_index = start_col; col_index < end_col; ++col_index) {
                    output_array[row_index * cols + col_index] = (float)mantissa[row_index * cols + col_index] * exponent_scale;
                }
            }
        }
    }
};
#endif
END2763