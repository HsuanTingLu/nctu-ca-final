// clang-format off
#include "types.hpp"
#include <cstring>
#include <stdexcept>
// clang-format on

// entry

namespace utils {

inline uint8_t char_hash(char c) {
    // maps ACGT$
    switch (c) {
        case 'A':
            return 0U;
            break;
        case 'C':
            return 1U;
            break;
        case 'G':
            return 2U;
            break;
        case 'T':
            return 3U;
            break;
        case '$':
            return 4U;
            break;
        default:
            throw std::invalid_argument("argument out of range");
            return UINT8_MAX;
            break;
    }
}

}  // namespace utils

entry::entry() {
    // Default constructor
}

entry::entry(const char* string) {
    for (unsigned int i = 0; i != 32; ++i) {
        this->array[i] = utils::char_hash(string[i << 1]) * 5U +
                         utils::char_hash(string[(i << 1) + 1U]);
    }

    // fills in '$$' at the end
    this->array[32] = utils::char_hash('$') * 5U + utils::char_hash('$');
}

entry& entry::operator=(const entry& other) {
    if (&other == this) {  // Check for self-assignment
        return *this;
    }
    // Reuse storage
    memmove(this->array, other.array, sizeof(uint8_t) * 33);
    return *this;
}

bool entry::operator>(const entry& other) const {
    // TODO: use CUDA SIMD
    return !(this->operator<(other));
}

bool entry::operator<(const entry& other) const {
    // TODO: use CUDA SIMD
    for (int i = 0; i != 33; ++i) {
        if (this->array[i] < other.array[i]) {
            return true;
        } else if (this->array[i] > other.array[i]) {
            return false;
        }
    }
    return false;
}

// entry_repr

entry_repr& entry_repr::operator=(const entry_repr& other) {
    // Check for self-assignment
    if (&other == this) {
        return *this;
    }
    this->str_idx = other.str_idx;
    this->str_shift = other.str_shift;
    this->char_shift = other.char_shift;
    return *this;
}
