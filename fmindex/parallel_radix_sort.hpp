/*
 * CPU-only multi-thread radix sort
 * Copyright (C) 2019  Hsuan-Ting Lu
 *
 * GNU General Public License v3.0+
 * (see LICENSE or https://www.gnu.org/licenses/)
 */

#ifndef PARALLEL_RADIX_SORT_HPP_
#define PARALLEL_RADIX_SORT_HPP_

// clang-format off
#include "types.hpp"
// clang-format on

namespace sort {
// 4 char, 4 Bytes, 32 bits
constexpr const unsigned int RADIX_BITS = 4 * 8;
// 4 char, 4 Bytes, 4 dna chars, 5^4 = 625
constexpr const unsigned int RADIX_SIZE = 625;
// 65-char = preliminary 1-char + 16 levels * 4 char
constexpr const unsigned int RADIX_LEVELS = 16;
//
constexpr const unsigned int PARTITION_CHARS =
    65 - (RADIX_BITS / 8) * RADIX_LEVELS;
// PARTITION_SIZE = power(1, PARTITION_CHARS)
constexpr const unsigned int PARTITION_SIZE = 5;
// 4 dna chars
constexpr const uint32_t RADIX_MASK = 0xffff;

// TODO: extern void insertionSort(std::vector<char*>);

void expand_rotation(const int array_size,
                     entry_repr* repr_array);

void partitioning(entry_repr*& repr_array, const unsigned int repr_array_size,
                  unsigned int partition_freq[PARTITION_SIZE]);

void radix_sort(entry_repr* repr_array, const unsigned int repr_array_size);

}  // namespace sort

#endif
