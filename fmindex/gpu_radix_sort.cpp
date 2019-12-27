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

#include "gpu_radix_sort.hpp"
// clang-format on
// DEBUG:
#define RED(x) "\033[31m" x "\033[0m"
#define GREEN(x) "\033[32m" x "\033[0m"
#define YELLOW(x) "\033[33m" x "\033[0m"

namespace sort {

void expand_rotation(const int array_size, entry_repr* repr_array) {
    /* Expands and creates the entire table of representations of
     * strings-to-be-sorted
     *
     * NOTE: cannot use async-task-launch because TA's machines sucks,
     *  cannot support that much thread per process
     */
    for (int str_idx = 0; str_idx != array_size; ++str_idx) {
        // Splitted one loop into two in case of needing parallelisation
        int repr_counter = str_idx * 65;
        for (int str_shift = 0; str_shift != 65; ++str_shift) {
            repr_array[repr_counter].str_idx = str_idx;
            repr_array[repr_counter].str_shift = str_shift;
            ++repr_counter;
        }
    }
}  // namespace sort

void partitioning(entry_repr*& repr_array, const unsigned int repr_array_size,
                  unsigned int partition_freq[PARTITION_SIZE]) {
    // Count bucket-key histogram to determine bucket sizes beforehand
    // TODO: counting is highly parallel
    uint32_t* bucket_key_label = new uint32_t[repr_array_size];
    for (unsigned int repr_idx = 0; repr_idx != repr_array_size; ++repr_idx) {
        // Extract partition bits (aka first char of string after rotation)
        entry_repr repr = repr_array[repr_idx];
        uint8_t* string = (repr.origin + repr.str_idx)->data;
        uint8_t bucket_idx = string[repr.str_shift];
        // Log number of bucket key occurrence of each entry
        bucket_key_label[repr_idx] = partition_freq[bucket_idx];
        // Update bucket key frequency info
        partition_freq[bucket_idx] += 1;
    }

    // temporary working area
    entry_repr* alt_array = new entry_repr[repr_array_size];
    //  Initialize bucket boundaries
    entry_repr* bucket_ptrs[PARTITION_SIZE];  // keeps track of bucket HEADs
    entry_repr* next = alt_array;
    for (unsigned int bucket_idx = 0; bucket_idx != PARTITION_SIZE;
         ++bucket_idx) {
        bucket_ptrs[bucket_idx] = next;
        next += partition_freq[bucket_idx];
    }
    // DEBUG: sanity check
    if (next != (alt_array + repr_array_size)) {
        throw std::logic_error(
            "partitioning:: final ptr should be exactly at the end of the "
            "alt_array");
    }

    // Partition (move)
    for (unsigned int repr_idx = 0; repr_idx != repr_array_size; ++repr_idx) {
        // work[repr_idx] = std::async(std::launch::async, [repr_array,
        // repr_idx, &bucket_mutex]() -> void {
        // Extract substring and categorize into bucket
        entry_repr repr = repr_array[repr_idx];
        uint8_t* string = (repr.origin + repr.str_idx)->data;
        unsigned int bucket_idx = static_cast<unsigned int>(string[repr.str_shift]);

        *(bucket_ptrs[bucket_idx] + bucket_key_label[repr_idx]) = repr;
        //});
    }

    // swap the entire repr array
    delete[] repr_array;
    repr_array = alt_array;
}

void radix_sort(entry_repr* repr_array, const unsigned int repr_array_size) {
    // alternate working area
    entry_repr* alt_array = new entry_repr[repr_array_size];
    entry_repr *from = repr_array,
               *to = alt_array;  // alternation pointers

    for (unsigned int pass = 0; pass != RADIX_LEVELS; ++pass) {
        //std::cerr << "---\npass: " << pass << " starts\n";
        // Count entry histograms to deteremine bucket sizes beforehand
        unsigned int frequency[RADIX_SIZE] = {0U};
        unsigned int* bucket_key_label = new unsigned int[repr_array_size];
        for (unsigned int i = 0; i != repr_array_size; ++i) {  // init array
            bucket_key_label[i] = 0U;
        }
        //std::cerr << "Creating data histogram\n";
        for (unsigned int repr_idx = 0; repr_idx != repr_array_size;
             ++repr_idx) {
            // Extract partition bits
            entry_repr repr = from[repr_idx];
            uint8_t* string = (repr.origin[repr.str_idx]).data;
            uint8_t partition_bits[4];
            //std::cerr << "\nstring: " << repr << "\n";
            unsigned int actual_shift =
                static_cast<unsigned int>(repr.str_shift) + 64 - 3 - pass * 4;
            actual_shift %= 65;
            //std::cerr << "shift: " << actual_shift << "\n";

            // clang-format off
            if ((actual_shift + 3) > 64) {
                // cyclic combination
                //DEBUG: std::cerr << YELLOW("CPY: ") << "data+" << actual_shift << ", size " << (65 - actual_shift) << "\n";
                //DEBUG: std::cerr << YELLOW("CPY: ") << "data+" << (65 - actual_shift) << ", size " << (4 - (65 - actual_shift)) << "\n";
                std::memcpy(partition_bits,
                            string + actual_shift,
                            (65 - actual_shift) * sizeof(uint8_t));
                std::memcpy(
                    partition_bits + (65 - actual_shift),
                    string,
                    (4 - (65 - actual_shift)) * sizeof(uint8_t));
            } else {
                // normal
                //DEBUG: std::cerr << YELLOW("CPY: ") << "data+" << actual_shift << ", size 4\n";
                std::memcpy(partition_bits,
                            string + actual_shift,
                            4 * sizeof(uint8_t));
            }
            // clang-format on

            /*DEBUG: std::cerr << "bits " << static_cast<unsigned
               int>(partition_bits[0])
                      << "," << static_cast<unsigned int>(partition_bits[1])
                      << "," << static_cast<unsigned int>(partition_bits[2])
                      << "," << static_cast<unsigned int>(partition_bits[3])
                      << "\n";*/
            unsigned int bucket_idx =
                static_cast<unsigned int>(partition_bits[0]) * 125 +
                static_cast<unsigned int>(partition_bits[1]) * 25 +
                static_cast<unsigned int>(partition_bits[2]) * 5 +
                static_cast<unsigned int>(partition_bits[3]);

            // Log number of bucket key occurrence of each entry
            bucket_key_label[repr_idx] = frequency[bucket_idx];
            // Update bucket key frequency info
            frequency[bucket_idx] += 1;

            /*std::cerr << "\nbucket_idx " << bucket_idx << "\n";
            std::cerr << RED("bucket_key_label ")
                      << static_cast<unsigned int>(bucket_key_label[repr_idx])
                      << "\n";
            std::cerr << YELLOW("frequency ")
                      << static_cast<unsigned int>(frequency[bucket_idx])
                      << "\n";*/
        }

        //std::cerr << "Init bucket boundaries\n";
        // Initialize bucket boundaries
        entry_repr* bucket_ptrs[RADIX_SIZE];
        entry_repr* next = to;
        for (unsigned int bucket_idx = 0; bucket_idx != RADIX_SIZE;
             ++bucket_idx) {
            bucket_ptrs[bucket_idx] = next;
            next += frequency[bucket_idx];
        }
        // DEBUG: sanity check
        if (next != (to + repr_array_size)) {
            std::cerr << "sanity failure on pass: " << pass << "\n";
            throw std::logic_error(
                "radix_sort:: final ptr should be exactly at the end of "
                "the "
                "alt_array");
        }

        //std::cerr << GREEN("actually moving data\n");
        // actually moving data to buckets
        for (unsigned int repr_idx = 0; repr_idx != repr_array_size;
             ++repr_idx) {
            // work[repr_idx] = std::async(std::launch::async, [from, repr_idx,
            // &bucket_mutex, &bucket_ptrs]() -> void {
            entry_repr repr = from[repr_idx];
            uint8_t* string = (repr.origin + repr.str_idx)->data;
            uint8_t partition_bits[4];
            unsigned int actual_shift =
                static_cast<unsigned int>(repr.str_shift) + 64 - 3 - pass * 4;
            actual_shift %= 65;

            // extract substring and categorize into bucket
            // clang-format off
            if ((actual_shift + 3) > 64) {
                // cyclic combination
                std::memcpy(partition_bits,
                            string + actual_shift,
                            (65 - actual_shift) * sizeof(uint8_t));
                std::memcpy(
                    partition_bits + (65 - actual_shift),
                    string,
                    (4 - (65 - actual_shift)) * sizeof(uint8_t));
            } else {
                // normal
                std::memcpy(partition_bits,
                            string + actual_shift,
                            4 * sizeof(uint8_t));
            }
            /*std::cerr << repr << "\n";
            std::cerr << PARTITION_CHARS + pass * 4 << " - " << PARTITION_CHARS + pass * 4 + 3 << "\n";
            std::cerr << RED("bits: ")
            << (unsigned int)(partition_bits[0])
            << " " << (unsigned int)(partition_bits[1])
            << " " << (unsigned int)(partition_bits[2])
            << " " << (unsigned int)(partition_bits[3]) << "\n";
            std::cerr << YELLOW("char: ")
            << rchar(partition_bits[0])
            << " " << rchar(partition_bits[1])
            << " " << rchar(partition_bits[2])
            << " " << rchar(partition_bits[3]) << "\n";*/
            // clang-format on

            unsigned int bucket_idx =
                static_cast<unsigned int>(partition_bits[0]) * 125 +
                static_cast<unsigned int>(partition_bits[1]) * 25 +
                static_cast<unsigned int>(partition_bits[2]) * 5 +
                static_cast<unsigned int>(partition_bits[3]);

            *(bucket_ptrs[bucket_idx] + bucket_key_label[repr_idx]) = repr;
            //});
        }

        // swap arrays (via pointers {from}/{to} swapping)
        entry_repr* ptr_swap_tmp = from;
        from = to;
        to = ptr_swap_tmp;
        //std::cerr << "pass: " << pass << " ends\n";
    }

    // return the correct array if ${RADIX_LEVELS} is odd
    if (RADIX_LEVELS & 1) {
        std::memcpy(repr_array, alt_array, repr_array_size);
        delete[] alt_array;
    } else {
        delete[] alt_array;
    }
}

}  // namespace sort
