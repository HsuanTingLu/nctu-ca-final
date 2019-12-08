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
inline uint8_t charpair_hash(char a, char b);
}

class entry {
    /* a string entry
     * structure: (66 characters in 33-char -> 33 Bytes)
     *   {64-character string} + {2-character $$ ending sequence}
     */
   public:
    entry();
    explicit entry(const char* string);
    entry& operator=(const entry& other);
    bool operator>(const entry& other) const;
    bool operator<(const entry& other) const;
    // TODO: stringify function

   public:
    uint8_t array[33];
};

struct entry_repr {
    /* represents a string entry,
     * in a condense form
     */
   public:
    entry get_entry();
    uint8_t* get_sub_entry(unsigned int str_idx, uint8_t str_shift, uint8_t char_shift);
   public:
    entry* str_idx;
    uint8_t str_shift : 7;   // 6-bit needed
    uint8_t char_shift : 1;  // 1-bit needed
};

#endif
