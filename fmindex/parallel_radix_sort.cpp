/*
 * CPU-only multi-thread radix sort
 * Copyright (C) 2019  Hsuan-Ting Lu
 *
 * GNU General Public License v3.0+
 * (see LICENSE or https://www.gnu.org/licenses/)
 */

// clang-format off
#include <cstdlib>
#include <cstring>

#include <memory>
#include <future>

#include "parallel_radix_sort.hpp"
// clang-format on

namespace sort {

void expand_rotation(entry* array, const int array_size,
                     entry_repr* repr_array) {
    /* Expands and creates the entire table of representations of
     * strings-to-be-sorted
     */
    for (int str_idx = 0; str_idx != array_size; ++str_idx) {
        auto work = std::async(
            std::launch::async, [&array, &repr_array, str_idx]() -> void {
                int repr_counter = str_idx * 65;
                for (int str_shift = 0; str_shift != 65; ++str_shift) {
                    repr_array[repr_counter].str_idx = &(array[str_idx]);
                    repr_array[repr_counter].str_shift = str_shift;
                    ++repr_counter;
                }
            });
        work.wait();
    }
}

void count_frequency(entry_repr* repr_array, const int repr_array_size,
                     unsigned int partition_freq[PARTITION_SIZE],
                     unsigned int frequency[RADIX_LEVELS][RADIX_SIZE]) {
    /* Counts the number of values that will fall into each bucket at each pass
     * so that it can be sized appropriately, aka taking the histogram of the
     * data
     */
    for (int i = 0; i != repr_array_size; ++i) {
        auto work = std::async(
            std::launch::async,
            [&repr_array, &partition_freq, &frequency, i]() -> void {
                entry_repr repr = repr_array[i];

                // extract full string
                uint8_t tmp[65];
                std::memcpy(tmp, repr.str_idx->data + repr.str_shift,
                            (65 - repr.str_shift) * sizeof(uint8_t));
                std::memcpy(tmp + 65 - repr.str_shift, repr.str_idx->data,
                            (repr.str_shift) * sizeof(uint8_t));

                // partitioning pass
                /* DEBUG:
                std::cerr << "@@@ " << utils::reverse_char(tmp[0]) << " "
                          << utils::reverse_char(tmp[1]) << " "
                          << utils::reverse_char(tmp[2]) << " "
                          << utils::reverse_char(tmp[3]) << " "
                          << utils::reverse_char(tmp[4]) << " -> " << repr << "
                @@@\n"; std::cerr << "                 "; for (uint8_t i = 0; i
                != 65; ++i) { std::cerr << utils::reverse_char(tmp[i]);
                }
                std::cerr << " <-\n";
                */
                partition_freq[static_cast<unsigned int>(tmp[0]) * 625 +
                               static_cast<unsigned int>(tmp[1]) * 125 +
                               static_cast<unsigned int>(tmp[2]) * 25 +
                               static_cast<unsigned int>(tmp[3]) * 5 +
                               static_cast<unsigned int>(tmp[4])]++;
                // radix pass
                for (unsigned int pass = 0; pass != RADIX_LEVELS; ++pass) {
                    unsigned int char_idx = 5 + pass * 4;
                    frequency[pass]
                             [static_cast<unsigned int>(tmp[char_idx + 0]) *
                                  125 +
                              static_cast<unsigned int>(tmp[char_idx + 1]) *
                                  25 +
                              static_cast<unsigned int>(tmp[char_idx + 2]) * 5 +
                              static_cast<unsigned int>(tmp[char_idx + 3])]++;
                }
            });
    }
}

void partitioning(entry_repr*& repr_array, const unsigned int repr_array_size,
                  unsigned int partition_freq[sort::PARTITION_SIZE]) {
    // init the bucket boundaries
    entry_repr* alt_array = static_cast<entry_repr*>(
        std::malloc(repr_array_size * sizeof(entry_repr)));

    entry_repr* bucket_ptrs[PARTITION_SIZE];
    entry_repr* next = alt_array;
    for (unsigned int i = 0; i != PARTITION_SIZE; ++i) {
        bucket_ptrs[i] = next;
        next += partition_freq[i];
    }

    // DEBUG: sanity check
    if (next != (alt_array + repr_array_size)) {
        throw std::logic_error(
            "partitioning:: final ptr should be exactly at the end of the "
            "alt_array");
    }

    // Partition (move)
    uint8_t tmp[5];
    for (unsigned int repr_idx = 0; repr_idx != repr_array_size; ++repr_idx) {
        entry_repr repr = repr_array[repr_idx];

        // extract substring and categorize into bucket
        if (repr.str_shift + 5 > 65) {
            // cyclic combination
            std::memcpy(tmp, repr.str_idx->data + repr.str_shift,
                        (65 - repr.str_shift) * sizeof(uint8_t));
            std::memcpy(tmp + 65 - repr.str_shift, repr.str_idx->data,
                        (repr.str_shift + 5 - 65) * sizeof(uint8_t));
        } else {
            // normal
            std::memcpy(tmp, repr.str_idx->data + repr.str_shift,
                        5 * sizeof(uint8_t));
        }

        unsigned int bucket_idx = static_cast<unsigned int>(tmp[0]) * 625 +
                                  static_cast<unsigned int>(tmp[1]) * 125 +
                                  static_cast<unsigned int>(tmp[2]) * 25 +
                                  static_cast<unsigned int>(tmp[3]) * 5 +
                                  static_cast<unsigned int>(tmp[4]);
        *bucket_ptrs[bucket_idx]++ = repr;
    }

    // swap the entire repr array
    std::free(repr_array);
    repr_array = alt_array;
}

void radix_sort(entry_repr*& repr_array, const unsigned int repr_array_size,
                unsigned int frequency[sort::RADIX_LEVELS][sort::RADIX_SIZE]) {
    entry_repr* alt_array = static_cast<entry_repr*>(
        std::malloc(repr_array_size * sizeof(entry_repr)));
    entry_repr *from = repr_array,
               *to = alt_array;  // alternation pointers

    for (unsigned int pass = 0; pass != RADIX_LEVELS; ++pass) {
        // init the bucket boundaries
        entry_repr* bucket_ptrs[RADIX_SIZE];
        entry_repr* next = to;
        for (unsigned int i = 0; i != RADIX_SIZE; ++i) {
            bucket_ptrs[i] = next;
            next += frequency[pass][i];
        }

        // DEBUG: sanity check
        if (next != (to + repr_array_size)) {
            throw std::logic_error(
                "radix_sort:: final ptr should be exactly at the end of the "
                "alt_array");
        }

        uint8_t tmp[4];
        for (unsigned int repr_idx = 0; repr_idx != repr_array_size;
             ++repr_idx) {
            entry_repr repr = from[repr_idx];

            // extract substring and categorize into bucket
            if (repr.str_shift + 4 > 65) {
                // cyclic combination
                std::memcpy(tmp, repr.str_idx->data + repr.str_shift,
                            (65 - repr.str_shift) * sizeof(uint8_t));
                std::memcpy(tmp + 65 - repr.str_shift, repr.str_idx->data,
                            (repr.str_shift + 4 - 65) * sizeof(uint8_t));
            } else {
                // normal
                std::memcpy(tmp, repr.str_idx->data + repr.str_shift,
                            4 * sizeof(uint8_t));
            }

            unsigned int bucket_idx = static_cast<unsigned int>(tmp[0]) * 125 +
                                      static_cast<unsigned int>(tmp[1]) * 25 +
                                      static_cast<unsigned int>(tmp[2]) * 5 +
                                      static_cast<unsigned int>(tmp[3]);
            *bucket_ptrs[bucket_idx]++ = repr;
        }

        // swap pointers
        entry_repr* ptr_swap_tmp = from;
        from = to;
        to = ptr_swap_tmp;
    }

    // return the correct array if ${RADIX_LEVELS} is odd
    if (RADIX_LEVELS & 1) {
        // swap the entire repr array
        std::free(repr_array);
        repr_array = alt_array;
    } else {
        std::free(alt_array);
    }
}

}  // namespace sort
