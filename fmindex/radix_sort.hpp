#ifndef RADIX_SORT_HPP_
#define RADIX_SORT_HPP_

// clang-format off
#include "types.hpp"
// clang-format on

namespace sort {
// 2 uint8_t, 2 Bytes, 16 bits
constexpr const unsigned int RADIX_BITS = 8 * 2;
// 2 uint8_t, 2 Bytes, 4 dna chars, 5^4 = 625
constexpr const unsigned int RADIX_SIZE = 625;
// 66-char = preliminary 6-char + 15 levels * 4 char
constexpr const unsigned int RADIX_LEVELS = 15;
// 4 dna chars
constexpr const uint32_t RADIX_MASK = 0xffff;

// TODO: extern void insertionSort(std::vector<char*>);

namespace SingleThread {

void expand_rotation(entry* array, const int array_size, entry_repr* repr_array,
                     const int repr_array_size);

void count_frequency(entry_repr* repr_array, const unsigned int size,
                     unsigned int freqency[RADIX_LEVELS][RADIX_SIZE]);

void radix_sort(entry_repr* repr_array, const unsigned int size);

}  // namespace SingleThread

namespace MultiThread {

void radix_sort();

}  // namespace MultiThread

namespace GPU {

void radix_sort();

}

}  // namespace sort
#endif
