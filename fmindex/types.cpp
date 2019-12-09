// clang-format off
#include <cstring>
#include <stdexcept>
#include <cstdio>  // DEBUG:

#include <iostream>  // DEBUG:

#include "types.hpp"
// clang-format on

// entry

namespace utils {

inline int8_t char_hash(char c) {
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
            throw std::invalid_argument("char_hash: argument out of range");
            return INT8_MAX;
    }
}

inline char reverse_char(int8_t c) {
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
            throw std::invalid_argument("reverse_char: argument out of range");
    }
}

}  // namespace utils

entry::entry() {
    // Default constructor
}

entry::entry(const char* string) {
    printf("%.65s\n", string);  // DEBUG:
    for (int char_idx = 0; char_idx != 65; ++char_idx) {
        this->data[char_idx] = utils::char_hash(string[char_idx]);
    }
}

entry& entry::operator=(const entry& other) {
    if (&other == this) {  // Check for self-assignment
        return *this;
    }
    // Reuse storage
    memmove(this->data, other.data, sizeof(int8_t) * 65);
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
    /*
    char tmp[65];
    memcpy(tmp, self.str_idx + self.str_shift, 65 - self.str_shift);  // left
    section memcpy(tmp + 65 - self.str_shift, self.str_idx, self.str_shift);  //
    right section
    */
    for (auto i = self.str_shift; i != 65; ++i) {
        os << utils::reverse_char(self.str_idx->data[i]);
    }
    for (auto i = 0; i != self.str_shift; ++i) {
        os << utils::reverse_char(self.str_idx->data[i]);
    }
    return os;
}
