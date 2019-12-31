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
// 8 char, 8 Bytes, 8 dna chars, 5^8 = 78125
constexpr const unsigned int RADIX_SIZE = 390625;
// 64-char = preliminary 1-char and (overlapped) 8 levels * 8 char
constexpr const unsigned int RADIX_LEVELS = 8;
// 1 char partitioning
constexpr const unsigned int PARTITION_CHARS = 1;
// PARTITION_SIZE = power(2, PARTITION_CHARS)
constexpr const unsigned int PARTITION_SIZE = 5;

void expand_rotation(const int array_size, entry_repr* repr_array);

void partitioning(entry_repr*& repr_array, const unsigned int repr_array_size,
                  unsigned int partition_freq[PARTITION_SIZE]);

void radix_sort(entry_repr* repr_array, const unsigned int repr_array_size);

}  // namespace sort

#endif
