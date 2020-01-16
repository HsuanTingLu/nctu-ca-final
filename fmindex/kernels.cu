/*
 * All CUDA kernels
 * Copyright (C) 2019  Hsuan-Ting Lu
 *
 * GNU General Public License v3.0+
 * (see LICENSE or https://www.gnu.org/licenses/)
 */

#ifndef magnificent_kernel_cu
#define magnificent_kernel_cu

#include "stdint.h"
#include "types.hpp"

__global__ void calc_bucket_index(unsigned int pass, entry* entry_array,
                                  entry_repr* repr_array,
                                  unsigned int array_size,
                                  unsigned int* bucket_indexes) {
    const unsigned int repr_idx =
        (static_cast<unsigned int>(blockIdx.x) << 10) +
        static_cast<unsigned int>(threadIdx.x);
    if (repr_idx > array_size) {
        return;
    }

    // TODO: extract
    const entry_repr repr = repr_array[repr_idx];
    const uint8_t* string = (entry_array[repr.str_idx]).data;
    uint8_t partition_bits[8];
    const unsigned int actual_shift =
        (static_cast<unsigned int>(repr.str_shift) + 64 - 8 - pass * 8) % 64;
    if ((actual_shift + 7) > 63) {
        // cyclic combination
        memcpy(partition_bits, string + actual_shift,
               (64 - actual_shift) * sizeof(uint8_t));
        memcpy(partition_bits + (64 - actual_shift), string,
               (8 - (64 - actual_shift)) * sizeof(uint8_t));
    } else {
        // normal
        memcpy(partition_bits, string + actual_shift, 8 * sizeof(uint8_t));
    }

    // TODO: calculate bucket_index
    const unsigned int bucket_index =
        static_cast<unsigned int>(partition_bits[0]) * 78125 +
        static_cast<unsigned int>(partition_bits[1]) * 15625 +
        static_cast<unsigned int>(partition_bits[2]) * 3125 +
        static_cast<unsigned int>(partition_bits[3]) * 625 +
        static_cast<unsigned int>(partition_bits[4]) * 125 +
        static_cast<unsigned int>(partition_bits[5]) * 25 +
        static_cast<unsigned int>(partition_bits[6]) * 5 +
        static_cast<unsigned int>(partition_bits[7]);

    bucket_indexes[repr_idx] = bucket_index;
}

// Build data histogram with sequential CPU code

__global__ void move_to_buckets(entry_repr* from, entry_repr* to,
                                unsigned int array_size,
                                unsigned int* bucket_HEADs,
                                unsigned int* bucket_key_label,
                                unsigned int* bucket_indexes) {
    const unsigned int repr_idx =
        (static_cast<unsigned int>(blockIdx.x) << 10) +
        static_cast<unsigned int>(threadIdx.x);
    if (repr_idx > array_size) {
        return;
    }
    const unsigned int bucket_index = bucket_indexes[repr_idx];

    to[bucket_HEADs[bucket_index] + bucket_key_label[repr_idx]] =
        from[repr_idx];
}

__global__ void expand_and_encode(entry* entry_array,
                                  entry_repr* repr_array,
                                  unsigned int array_size,
                                  char (*result_array)[32]) {
    const unsigned int repr_idx =
        (static_cast<unsigned int>(blockIdx.x) << 10) +
        static_cast<unsigned int>(threadIdx.x);
    if (repr_idx > array_size) {
        return;
    }
    const entry_repr repr = repr_array[repr_idx];
    const uint8_t* string = (entry_array[repr.str_idx]).data;

    // TODO: extract full entry with shift
    uint8_t extraction[64];
    memcpy(extraction + repr.str_shift, string, (64 - repr.str_shift) * sizeof(uint8_t));
    memcpy(extraction, string + (64 - repr.str_shift), repr.str_shift * sizeof(uint8_t));

    // TODO: squeeze them into char[32]
    for(int i = 0 ; i < 32 ; ++i ){
        result_array[repr_idx][i] = static_cast<char>((extraction[2 * i] << 4) + extraction[2 * i + 1]); 
    }
}

#endif