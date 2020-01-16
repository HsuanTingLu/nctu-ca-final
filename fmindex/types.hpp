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

__host__ __device__ uint8_t char_hash(char c);
__host__ __device__ char reverse_char(uint8_t c);
__host__ __device__ unsigned int four_bit_encode(char c);

}  // namespace utils

class entry {
    /* a string entry
     * structure: (66 characters in 33-char -> 33 Bytes)
     *   {64-character string} + {2-character $$ ending sequence}
     */
    friend std::ostream& operator<<(std::ostream& os, entry& self);

   public:
    // TODO: re-implement explicit constructors
    __host__ __device__ entry();
    __host__ __device__ entry(const char* string);
    __host__ __device__ entry& operator=(const entry& other);

   public:
    uint8_t data[64];
};

class entry_repr {
    /* represents a string entry,
     * in a condense form
     */
    friend std::ostream& operator<<(std::ostream& os, entry_repr& self);

   public:
    static entry* origin;

   public:
    // TODO: re-implement explicit constructors
    __host__ __device__ entry_repr();
    __host__ __device__ entry_repr(uint32_t str_idx, uint8_t str_shift);
    __host__ __device__
    entry_repr(const entry_repr& other);  // copy constructor
    __host__ __device__ entry_repr& operator=(
        const entry_repr& other);  // copy assignment

   public:
    uint32_t str_idx : 24;
    uint8_t str_shift : 8;
};

#endif
