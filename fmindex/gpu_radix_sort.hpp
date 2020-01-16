/*
 * GPU radix sort
 * Copyright (C) 2019  Hsuan-Ting Lu
 *
 * GNU General Public License v3.0+
 * (see LICENSE or https://www.gnu.org/licenses/)
 */

#ifndef GPU_RADIX_SORT_HPP_
#define GPU_RADIX_SORT_HPP_

// clang-format off
#include "types.hpp"
// clang-format on

namespace sort {
// 8 char, 8 Bytes, 8 dna chars, 5^8 = 390625
constexpr const unsigned int RADIX_SIZE = 390625;
// 64-char = preliminary 1-char and (overlapped) 8 levels * 8 char
constexpr const unsigned int RADIX_LEVELS = 8;
// 1 char partitioning
constexpr const unsigned int PARTITION_CHARS = 1;
// PARTITION_SIZE = power(2, PARTITION_CHARS)
constexpr const unsigned int PARTITION_SIZE = 5;

void expand_rotation(const int array_size, entry_repr* repr_array);

void radix_sort(entry_repr* repr_array, const unsigned int entry_array_size);

void encode(entry* entry_array,
            entry_repr* repr_array,
            unsigned int repr_array_size,
            char (*result_array)[32]);

}  // namespace sort

#endif
