// clang-format off
#include <cstdlib>

#include <memory>

#include "radix_sort.hpp"
// clang-format on

namespace sort {

namespace SingleThread {

void expand_rotation(entry* array, const int array_size, entry_repr* repr_array,
                     const int repr_array_size) {
    /* Expands and creates the entire table of representations of
     * strings-to-be-sorted
     */
    int repr_counter = 0;  // this will become a concurrency bottleneck
    for (int str_idx = 0; str_idx != array_size; ++str_idx) {
        for (int str_shift = 0; str_shift != 65; ++str_shift) {
            repr_array[repr_counter].str_idx = &(array[str_idx]);
            repr_array[repr_counter].str_shift = str_shift;
            ++repr_counter;
        }
    }
    if (repr_counter != repr_array_size) {  // DEBUG:
        throw std::runtime_error("expanded-array size mismatch");
    }
}

void count_frequency(entry_repr* repr_array, const int repr_array_size,
                     unsigned int partition_freq[PARTITION_SIZE],
                     unsigned int freqency[RADIX_LEVELS][RADIX_SIZE]) {
    /* Counts the number of values that will fall into each bucket at each pass
     * so that it can be sized appropriately, aka taking the histogram of the
     * data
     */
    for (int i = 0; i != repr_array_size; ++i) {
        entry_repr repr = repr_array[i];

        // extract full string
        uint8_t tmp[65];
        memcpy(tmp, repr.str_idx->data + repr.str_shift,
               (65 - repr.str_shift) * sizeof(uint8_t));
        memcpy(tmp + 65 - repr.str_shift, repr.str_idx->data,
               (repr.str_shift) * sizeof(uint8_t));

        // partitioning pass
        /* DEBUG:
        std::cerr << "@@@ " << utils::reverse_char(tmp[0]) << " "
                  << utils::reverse_char(tmp[1]) << " "
                  << utils::reverse_char(tmp[2]) << " "
                  << utils::reverse_char(tmp[3]) << " "
                  << utils::reverse_char(tmp[4]) << " -> " << repr << " @@@\n";
        std::cerr << "                 ";
        for (uint8_t i = 0; i != 65; ++i) {
            std::cerr << utils::reverse_char(tmp[i]);
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
            freqency[pass][static_cast<unsigned int>(tmp[char_idx + 0]) * 125 +
                           static_cast<unsigned int>(tmp[char_idx + 1]) * 25 +
                           static_cast<unsigned int>(tmp[char_idx + 2]) * 5 +
                           static_cast<unsigned int>(tmp[char_idx + 3])]++;
        }
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
            memcpy(tmp, repr.str_idx->data + repr.str_shift,
                   65 - repr.str_shift);
            memcpy(tmp + 65 - repr.str_shift, repr.str_idx->data,
                   repr.str_shift + 5 - 65);
        } else {
            // normal
            memcpy(tmp, repr.str_idx->data + repr.str_shift, 5);
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

void radix_sort(entry_repr* repr_array, const unsigned int size) {
    std::unique_ptr<entry_repr[]> alt_array(new entry_repr[size]);
    entry_repr *from = repr_array,
               *to = alt_array.get();  // alternation pointers

    // Scan for distribution
    unsigned int freqency[RADIX_LEVELS][RADIX_SIZE] = {};

    for (int pass = 0; pass != RADIX_LEVELS; ++pass) {
    }
}

}  // namespace SingleThread

namespace MultiThread {

void radix_sort() {}

}  // namespace MultiThread

namespace GPU {

void radix_sort() {}

}  // namespace GPU

}  // namespace sort
