/* all major types used in the project
 *
 * injects directly into the global namespace
 */

#ifndef TYPES_HPP_
#define TYPES_HPP_

// clang-format off
#include <cstdint>

#include <ostream>
// clang-format on

namespace utils {

uint8_t char_hash(char c);
char reverse_char(uint8_t c);

}  // namespace utils

class entry {
    /* a string entry
     * structure: (66 characters in 33-char -> 33 Bytes)
     *   {64-character string} + {2-character $$ ending sequence}
     */
    friend std::ostream& operator<<(std::ostream& os, entry& self);

   public:
    explicit entry();
    explicit entry(const char* string);
    entry& operator=(const entry& other);

   public:
    uint8_t data[65];
};

class entry_repr {
    /* represents a string entry,
     * in a condense form
     */
    friend std::ostream& operator<<(std::ostream& os, entry_repr& self);

   public:
    static entry* origin;

   public:
    explicit entry_repr();
    entry_repr(uint32_t str_idx, uint8_t str_shift);
    entry_repr(const entry_repr& other);             // copy constructor
    entry_repr& operator=(const entry_repr& other);  // copy assignment

   public:
    uint32_t str_idx : 24;
    uint8_t str_shift : 8;
};

#endif
