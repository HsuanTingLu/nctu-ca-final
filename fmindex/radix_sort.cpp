// clang-format off
#include "radix_sort.hpp"
#include <memory>
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

void count_frequency(entry_repr* repr_array, const int size,
                     int freqency[RADIX_LEVELS][RADIX_SIZE]) {
    /* Counts the number of values that will fall into each bucket at each pass
     * so that it can be sized appropriately, aka taking the histogram of the
     * data
     */
    for (int i = 0; i != size; ++i) {
        // disregard top 6-char since they are trivial after partitioning
        auto value = repr_array[i];
        for (int pass = 0; pass != RADIX_LEVELS; ++pass) {
            freqency[pass][0];
        }
    }
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
