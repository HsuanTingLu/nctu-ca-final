#include "types.hpp"
#include <cstring>

// entry

namespace utils {

inline uint8_t char_hash(char c) {
    // maps ACGT$ to 01234
    switch (c) {
        case 'A':
            return 0;
            break;
        case 'C':
            return 1;
            break;
        case 'G':
            return 2;
            break;
        case 'T':
            return 3;
            break;
        case '$':
            return 4;
            break;
        default:
            return UINT8_MAX;
            break;
    }
}

}  // namespace utils

entry::entry() {
    // Default constructor
}

entry::entry(const char* string) {
    for (int i = 0; i != 32; ++i) {
        this->array[i] = (utils::char_hash(string[i << 1]) << 4) |
                         (0x0f & utils::char_hash(string[(i << 1) + 1]));
    }
    this->array[32] = (4 << 4) | (0x0f & 4);  // fills in '$$' at the end
}

entry& entry::operator=(const entry& other) {
    if(&other == this) {  // Check for self-assignment
        return *this;
    }
    // Reuse storage
    memmove(this->array, other.array, sizeof(uint8_t) * 33);
}

bool entry::operator>(const entry& other) const {
    return !(this->operator<(other));
}

bool entry::operator<(const entry& other) const {
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
