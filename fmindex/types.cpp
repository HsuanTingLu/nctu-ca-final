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
            return 0;
            break;
        case 'C':
            return 1;
            break;
        case 'G':
            return 2;
            break;
        case 'T':
            return 4;
            break;
        case '$':
            return 8;
            break;
        default:
            throw std::invalid_argument("argument out of range");
            return UINT8_MAX;
            break;
    }
}

inline uint8_t charpair_hash(uint8_t a, uint8_t b) {
    // maps char PAIRS to continous numbers
    switch ((a << 4) + b) {
        case ((0 << 4) + 0):  // AA
            return 0;
            break;
        case ((0 << 4) + 1):  // AC
            return 1;
            break;
        case ((0 << 4) + 2):  // AG
            return 2;
            break;
        case ((0 << 4) + 4):  // AT
            return 3;
            break;
        case ((0 << 4) + 8):  // A$
            return 4;
            break;
        case ((1 << 4) + 0):  // CA
            return 5;
            break;
        case ((1 << 4) + 1):  // CC
            return 6;
            break;
        case ((1 << 4) + 2):  // CG
            return 7;
            break;
        case ((1 << 4) + 4):  // CT
            return 8;
            break;
        case ((1 << 4) + 8):  // C$
            return 9;
            break;
        case ((2 << 4) + 0):  // GA
            return 10;
            break;
        case ((2 << 4) + 1):  // GC
            return 11;
            break;
        case ((2 << 4) + 2):  // GG
            return 12;
            break;
        case ((2 << 4) + 4):  // GT
            return 13;
            break;
        case ((2 << 4) + 8):  // G$
            return 14;
            break;
        case ((4 << 4) + 0):  // TA
            return 15;
            break;
        case ((4 << 4) + 1):  // TC
            return 16;
            break;
        case ((4 << 4) + 2):  // TG
            return 17;
            break;
        case ((4 << 4) + 4):  // TT
            return 18;
            break;
        case ((4 << 4) + 8):  // T$
            return 19;
            break;
        case ((8 << 4) + 0):  // $A
            return 20;
            break;
        case ((8 << 4) + 1):  // $C
            return 21;
            break;
        case ((8 << 4) + 2):  // $G
            return 22;
            break;
        case ((8 << 4) + 4):  // $T
            return 23;
            break;
        case ((8 << 4) + 8):  // $$
            return 24;
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
    for (int i = 0; i != 32; ++i) {
        // clang-format off
        this->array[i] = utils::charpair_hash(
            utils::char_hash(string[i << 1]),
            utils::char_hash(string[(i << 1) + 1]));
        // clang-format on
    }

    // fills in '$$' at the end
    this->array[32] =
        utils::charpair_hash(utils::char_hash('$'), utils::char_hash('$'));
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
