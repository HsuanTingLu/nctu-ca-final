// clang-format off
#include <cstring>
#include <stdexcept>

#include "types.hpp"
// clang-format on

// entry

namespace utils {

inline uint8_t char_hash(char c) {
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
            throw std::domain_error("char_hash: argument out of range");
    }
}

inline char reverse_char(uint8_t c) {
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
            throw std::domain_error("reverse_char: argument out of range");
    }
}

}  // namespace utils

entry::entry() {
    // Default constructor
}

entry::entry(const char* string) {
    for (int char_idx = 0; char_idx != 65; ++char_idx) {
        this->data[char_idx] = utils::char_hash(string[char_idx]);
    }
}

entry& entry::operator=(const entry& other) {
    if (&other == this) {  // Check for self-assignment
        return *this;
    }
    // Reuse storage
    std::memmove(this->data, other.data, sizeof(uint8_t) * 65);
    return *this;
}

std::ostream& operator<<(std::ostream& os, entry& self) {
    // pruned output (with only 1 $)
    for (int i = 0; i != 65; ++i) {
        os << utils::reverse_char(self.data[i]);
    }
    return os;
}

// entry_repr
entry_repr::entry_repr(entry* str_idx, uint8_t str_shift)
    : str_idx(str_idx), str_shift(str_shift) {}

entry_repr::entry_repr() : str_idx(nullptr), str_shift(0) {}

entry_repr::entry_repr(const entry_repr& other) {
    this->str_idx = other.str_idx;
    this->str_shift = other.str_shift;
}

entry_repr& entry_repr::operator=(const entry_repr& other) {
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
    uint8_t tmp[65];
    // left section
    std::memcpy(tmp, self.str_idx->data + self.str_shift,
                (65 - self.str_shift) * sizeof(uint8_t));
    // right section
    std::memcpy(tmp + 65 - self.str_shift, self.str_idx->data,
                self.str_shift * sizeof(uint8_t));

    for (uint8_t i = 0; i != 65; ++i) {
        os << utils::reverse_char(tmp[i]);
    }
    return os;
}
