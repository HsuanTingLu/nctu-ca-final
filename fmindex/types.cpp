// clang-format off
#include <cstring>
#include <stdexcept>

#include "types.hpp"
// clang-format on

// entry

namespace utils {

__host__ __device__ uint8_t char_hash(char c) {
    // maps ACGT$
    switch (c) {
        case '$':
            return 0;
            break;
        case 'A':
            return 1;
            break;
        case 'C':
            return 2;
            break;
        case 'G':
            return 3;
            break;
        case 'T':
            return 4;
            break;
        default:
            // HACK: device code does not support exception handling
            // throw std::domain_error("char_hash: argument out of range");
            return 7;
    }
}

__host__ __device__ char reverse_char(uint8_t c) {
    switch (c) {
        case 0:
            return '$';
            break;
        case 1:
            return 'A';
            break;
        case 2:
            return 'C';
            break;
        case 3:
            return 'G';
            break;
        case 4:
            return 'T';
            break;
        default:
            // HACK: device code does not support exception handling
            // throw std::domain_error("reverse_char: argument out of range");
            return 7;
    }
}

__host__ __device__ unsigned int four_bit_encode(char c){
	switch (c) {
        case '$':
            return 0;
            break;
        case 'A':
            return 1;
            break;
        case 'C':
            return 2;
            break;
        case 'G':
            return 4;
            break;
        case 'T':
            return 8;
            break;
        default:
            return 7;
    }
}

}  // namespace utils

__host__ __device__ entry::entry() {
    // Default constructor
}

__host__ __device__ entry::entry(const char* string) {
    for (int char_idx = 0; char_idx != 64; ++char_idx) {
        this->data[char_idx] = utils::char_hash(string[char_idx]);
    }
}

__host__ __device__ entry& entry::operator=(const entry& other) {
    if (&other == this) {  // Check for self-assignment
        return *this;
    }
    // Reuse storage
    std::memcpy(this->data, other.data, sizeof(uint8_t) * 64);
    return *this;
}

std::ostream& operator<<(std::ostream& os, entry& self) {
    // pruned output (with only 1 $)
    for (int i = 0; i != 64; ++i) {
        os << utils::reverse_char(self.data[i]);
    }
    return os;
}

// entry_repr
entry* entry_repr::origin;

__host__ __device__ entry_repr::entry_repr(uint32_t str_idx, uint8_t str_shift)
    : str_idx(str_idx), str_shift(str_shift) {}

__host__ __device__ entry_repr::entry_repr() : str_idx(0), str_shift(0) {}

__host__ __device__ entry_repr::entry_repr(const entry_repr& other) {
    this->str_idx = other.str_idx;
    this->str_shift = other.str_shift;
}

__host__ __device__ entry_repr& entry_repr::operator=(const entry_repr& other) {
    // Check for self-assignment
    if (&other == this) {
        return *this;
    }
    this->str_idx = other.str_idx;
    this->str_shift = other.str_shift;
    return *this;
}

std::ostream& operator<<(std::ostream& os, entry_repr& self) {
    // Cycle shift: amount=self.str_shift
    uint8_t tmp[64];
    uint8_t* string = (self.origin[self.str_idx]).data;
    // left section
    std::memcpy(tmp, string + self.str_shift,
                (64 - self.str_shift) * sizeof(uint8_t));
    // right section
    std::memcpy(tmp + 64 - self.str_shift, string,
                self.str_shift * sizeof(uint8_t));

    for (uint8_t i = 0; i != 64; ++i) {
        os << utils::reverse_char(tmp[i]);
    }
    return os;
}
