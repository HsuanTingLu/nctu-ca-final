/* all major types used in the project
 *
 * injects directly into the global namespace
 */

#ifndef TYPES_HPP_
#define TYPES_HPP_

// clang-format off
#include <cstdint>
// clang-format on

namespace utils {
inline uint8_t char_hash(char c);
}

class entry {
    /* a string entry
     * structure: (66 characters in 33-char -> 33 Bytes)
     *   {64-character string} + {2-character $$ ending sequence}
     */
   public:
    __device__ __host__ explicit entry(const char* const string);
    __host__ bool operator>(const entry& other) const;
    __host__ bool operator<(const entry& other) const;

   public:
    uint8_t array[33];
};

struct entry_repr {
    /* represents a string entry,
     * in a condense form
     */
   public:
    entry* str_idx;
    uint8_t str_shift : 7;  // 6-bit needed
    uint8_t char_shift : 1;  // 1-bit needed
};

#endif
