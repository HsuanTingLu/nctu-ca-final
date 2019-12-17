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
#include <mutex>
#include <iostream>  // DEBUG:

#include "parallel_radix_sort.hpp"
// clang-format on

namespace sort {

void expand_rotation(entry* array, const int array_size,
                     entry_repr* repr_array) {
    /* Expands and creates the entire table of representations of
     * strings-to-be-sorted
     */
    std::future<void>* work = new std::future<void>[array_size];
    for (int str_idx = 0; str_idx != array_size; ++str_idx) {
        work[str_idx] = std::async(
            std::launch::async, [&array, &repr_array, str_idx]() -> void {
                int repr_counter = str_idx * 65;
                for (int str_shift = 0; str_shift != 65; ++str_shift) {
                    repr_array[repr_counter].str_idx = &(array[str_idx]);
                    repr_array[repr_counter].str_shift = str_shift;
                    ++repr_counter;
                }
            });
    }
    for (int str_idx = 0; str_idx != array_size; ++str_idx) {
        work[str_idx].wait();
    }
    delete[] work;
}

void count_frequency(entry_repr* repr_array, const int repr_array_size,
                     unsigned int partition_freq[PARTITION_SIZE],
                     unsigned int frequency[RADIX_LEVELS][RADIX_SIZE]) {
    /* Counts the number of values that will fall into each bucket at each pass
     * so that it can be sized appropriately, aka taking the histogram of the
     * data
     */
    for (int repr_idx = 0; repr_idx != repr_array_size; ++repr_idx) {
        entry_repr repr = repr_array[repr_idx];

        // extract full string
        uint8_t tmp[65];
        std::memcpy(tmp, repr.str_idx->data + repr.str_shift,
                    (65 - repr.str_shift) * sizeof(uint8_t));
        std::memcpy(tmp + 65 - repr.str_shift, repr.str_idx->data,
                    (repr.str_shift) * sizeof(uint8_t));

        // partitioning pass
        partition_freq[static_cast<unsigned int>(tmp[0]) * 625 +
                       static_cast<unsigned int>(tmp[1]) * 125 +
                       static_cast<unsigned int>(tmp[2]) * 25 +
                       static_cast<unsigned int>(tmp[3]) * 5 +
                       static_cast<unsigned int>(tmp[4])] += 1;
        // radix pass
        for (unsigned int pass = 0; pass != RADIX_LEVELS; ++pass) {
            unsigned int char_idx = 5 + pass * 4;
            unsigned int freq_bucket_idx =
                static_cast<unsigned int>(tmp[char_idx + 0]) * 125 +
                static_cast<unsigned int>(tmp[char_idx + 1]) * 25 +
                static_cast<unsigned int>(tmp[char_idx + 2]) * 5 +
                static_cast<unsigned int>(tmp[char_idx + 3]);
            frequency[pass][freq_bucket_idx] += 1;
        }
    }
}  // namespace sort

void partitioning(entry_repr*& repr_array, const unsigned int repr_array_size,
                  unsigned int partition_freq[sort::PARTITION_SIZE]) {
    // bucket boundaries in array
    entry_repr* bucket_ptrs[PARTITION_SIZE];
    std::mutex bucket_mutex[PARTITION_SIZE];

    // temporary working area
    entry_repr* alt_array = new entry_repr[repr_array_size];
    entry_repr* next = alt_array;

    // initialize the bucket boundaries
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
    std::future<void>* work = new std::future<void>[repr_array_size];
    for (unsigned int repr_idx = 0; repr_idx != repr_array_size; ++repr_idx) {
        work[repr_idx] = std::async(
            std::launch::async,
            [repr_array, repr_idx, &bucket_mutex, &bucket_ptrs]() -> void {
                entry_repr repr = repr_array[repr_idx];
                uint8_t tmp[5];

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

                unsigned int bucket_idx =
                    static_cast<unsigned int>(tmp[0]) * 625 +
                    static_cast<unsigned int>(tmp[1]) * 125 +
                    static_cast<unsigned int>(tmp[2]) * 25 +
                    static_cast<unsigned int>(tmp[3]) * 5 +
                    static_cast<unsigned int>(tmp[4]);

                // Critical section (mutual exclusion block)
                {
                    const std::lock_guard<std::mutex> scoped_bucket_lock(
                        bucket_mutex[bucket_idx]);

                    // sequential version
                    // *bucket_ptrs[bucket_idx]++ = repr;

                    // atomic(X) fetch_add(O)
                    entry_repr* obj = bucket_ptrs[bucket_idx];
                    bucket_ptrs[bucket_idx] += 1;
                    // do op
                    *obj = repr;
                }
            });
    }
    for (unsigned int repr_idx = 0; repr_idx != repr_array_size; ++repr_idx) {
        work[repr_idx].wait();
    }
    delete[] work;

    // swap the entire repr array
    delete[] repr_array;
    repr_array = alt_array;
}

void radix_sort(entry_repr*& repr_array, const unsigned int repr_array_size,
                unsigned int frequency[sort::RADIX_LEVELS][sort::RADIX_SIZE]) {
    // temporary working area
    entry_repr* alt_array = new entry_repr[repr_array_size];
    entry_repr *from = repr_array,
               *to = alt_array;  // alternation pointers

    for (unsigned int pass = 0; pass != RADIX_LEVELS; ++pass) {
        std::cerr << "pass: " << pass << " starts\n";
        // bucket boundaries in array
        entry_repr* bucket_ptrs[RADIX_SIZE];
        std::mutex bucket_mutex[RADIX_SIZE];

        // initialize the bucket boundaries
        entry_repr* next = to;
        for (unsigned int i = 0; i != RADIX_SIZE; ++i) {
            bucket_ptrs[i] = next;
            next += frequency[pass][i];
        }

        // DEBUG: sanity check
        if (next != (to + repr_array_size)) {
            std::cerr << "sanity failure on pass: " << pass << "\n";
            throw std::logic_error(
                "radix_sort:: final ptr should be exactly at the end of "
                "the "
                "alt_array");
        }

        // actually moving data to buckets
        std::future<void>* work = new std::future<void>[repr_array_size];
        for (unsigned int repr_idx = 0; repr_idx != repr_array_size;
             ++repr_idx) {
            work[repr_idx] = std::async(
                std::launch::async,
                [from, repr_idx, &bucket_mutex, &bucket_ptrs]() -> void {
                    entry_repr repr = from[repr_idx];
                    uint8_t tmp[4];

                    // extract substring and categorize into bucket
                    if (repr.str_shift + 4 > 65) {
                        // cyclic combination
                        std::memcpy(tmp, repr.str_idx->data + repr.str_shift,
                                    (65 - repr.str_shift) * sizeof(uint8_t));
                        std::memcpy(
                            tmp + 65 - repr.str_shift, repr.str_idx->data,
                            (repr.str_shift + 4 - 65) * sizeof(uint8_t));
                    } else {
                        // normal
                        std::memcpy(tmp, repr.str_idx->data + repr.str_shift,
                                    4 * sizeof(uint8_t));
                    }

                    unsigned int bucket_idx =
                        static_cast<unsigned int>(tmp[0]) * 125 +
                        static_cast<unsigned int>(tmp[1]) * 25 +
                        static_cast<unsigned int>(tmp[2]) * 5 +
                        static_cast<unsigned int>(tmp[3]);
                    // Critical section (mutual exclusion block)
                    {
                        const std::lock_guard<std::mutex> scoped_bucket_lock(
                            bucket_mutex[bucket_idx]);

                        // sequential version
                        //*bucket_ptrs[bucket_idx]++ = repr;

                        // atomic(X) fetch_add(O)
                        entry_repr* obj = bucket_ptrs[bucket_idx];
                        bucket_ptrs[bucket_idx] += 1;
                        // do op
                        *obj = repr;
                    }
                });  // FIXME: impl async dispatch
        }

        for (unsigned int repr_idx = 0; repr_idx != repr_array_size;
             ++repr_idx) {
            work[repr_idx].wait();
        }
        delete[] work;

        // swap arrays (via pointers {from}/{to} swapping)
        entry_repr* ptr_swap_tmp = from;
        from = to;
        to = ptr_swap_tmp;
        std::cerr << "pass: " << pass << " ends\n";
    }

    // return the correct array if ${RADIX_LEVELS} is odd
    if (RADIX_LEVELS & 1) {
        // swap the entire repr array
        delete[] repr_array;
        repr_array = alt_array;
    } else {
        delete[] alt_array;
    }
}

}  // namespace sort
