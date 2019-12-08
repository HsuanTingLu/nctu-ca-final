// clang-format off
#include "radix_sort.hpp"
#include <memory>
// clang-format on

namespace sort {

namespace SingleThread {

void expand_rotation(entry* array, const int array_size, entry_repr* repr_array,
                     const int repr_array_size) {
    /* Expands and creates the entire table of representations of strings-to-be-sorted
     */
}

void count_frequency(entry_repr* repr_array, const unsigned int size,
                     unsigned int freqency[RADIX_LEVELS][RADIX_SIZE]) {
    /* Counts the number of values that will fall into each bucket at each pass
     * so that it can be sized appropriately, aka taking the histogram of the
     * data
     */
}

void radix_sort(entry_repr* repr_array, const unsigned int size) {


}

}  // namespace SingleThread

namespace MultiThread {


}  // namespace MultiThread

namespace GPU {


}

}  // namespace sort
