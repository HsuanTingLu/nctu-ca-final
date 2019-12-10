/*
 * Host side master program
 * Copyright (C) 2019  Hsuan-Ting Lu
 *
 * GNU General Public License v3.0+
 * (see LICENSE or https://www.gnu.org/licenses/)
 */

// clang-format off
#include <cstdio>
#include <cstring>

#include <fstream>
#include <iostream>
#include <ratio>
#include <memory>
#include <string>
#include <stdexcept>
#include <numeric>

#include "types.hpp"
#include "radix_sort.hpp"
// clang-format on

// TODO: pin memory
void read_input(std::ifstream* ifs, entry* array, const int INPUTSIZE) {
    char buffer[65];
    buffer[64] = '$';
    for (int str_idx = 0; str_idx != INPUTSIZE; ++str_idx) {
        ifs->read(buffer, 64);
        ifs->ignore();
        array[str_idx] = entry(buffer);
    }
}

int main(int argc, char** argv) {
    if (argc != 1 + 2) {  // DEBUG:
        throw std::invalid_argument("need 2 arguments");
    }

    const int INPUTSIZE = std::stoi(argv[2]);  // HACK: use "wc -l" for this
    const int EXPANDEDSIZE = 65 * INPUTSIZE;
    std::cerr << "expected output size :: str_array: " << INPUTSIZE
              << ", rotate_expand: " << EXPANDEDSIZE << "\n";
    entry* str_array;
    entry_repr* repr_array;

    // Read to memory
    std::ifstream ifs(argv[1], std::ifstream::in);
    if (ifs) {
        // TODO: Count number of lines of the input file, HACK: use "wc -l"

        // Allocate array
        str_array = static_cast<entry*>(malloc(INPUTSIZE * sizeof(entry)));
        repr_array =
            static_cast<entry_repr*>(malloc(EXPANDEDSIZE * sizeof(entry_repr)));

        // Read input
        read_input(&ifs, str_array, INPUTSIZE);
        ifs.close();
    } else {
        throw std::invalid_argument("Cannot open file");
    }

    std::cout << std::endl;

    for (int i = 0; i != INPUTSIZE; ++i) {
        // std::cout << str_array[i] << std::endl;
    }
    // std::cout << "\n";

    sort::SingleThread::expand_rotation(str_array, INPUTSIZE, repr_array,
                                        EXPANDEDSIZE);
    for (int i = 0; i != EXPANDEDSIZE; ++i) {
        if (!(i % 65)) {
            // std::cout << "< " << i / 65 << " >\n";
        }
        // std::cout << repr_array[i] << std::endl;
    }

    // Scan for distribution
    unsigned int partition_freq[sort::PARTITION_SIZE] = {};
    unsigned int freqency[sort::RADIX_LEVELS][sort::RADIX_SIZE] = {};
    sort::SingleThread::count_frequency(repr_array, EXPANDEDSIZE,
                                        partition_freq, freqency);

    std::cerr << "partition frequency:\n";
    std::cerr << "(sum = "
              << std::accumulate(partition_freq,
                                 partition_freq + sort::PARTITION_SIZE, 0)
              << ")\n";
    for (int i = 0; i != sort::PARTITION_SIZE; ++i) {
        // std::cerr << partition_freq[i] << " ";
        if (!(i % 80) && i != 0) {
            // std::cerr << "\n";
        }
    }
    // std::cerr << "\n";

    // Print data
    /*
    for (auto str : str_array) {
        // custom string print length :: READLENGTH + 1
        printf("%.65s\n", str);
    }
    */
    free(str_array);
    free(repr_array);
}
