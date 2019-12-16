/*
 * Host side master program
 * Copyright (C) 2019  Hsuan-Ting Lu
 *
 * GNU General Public License v3.0+
 * (see LICENSE or https://www.gnu.org/licenses/)
 */

// clang-format off
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <stdexcept>
#include <numeric>

#include "types.hpp"
#include "parallel_radix_sort.hpp"
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
        throw std::invalid_argument("2 arguments needed");
    }
    // HACK: use "wc -l" for file line counting
    const int INPUTSIZE = std::stoi(argv[2]);
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
        str_array = new entry[INPUTSIZE];
        repr_array = new entry_repr[EXPANDEDSIZE];

        // Read input
        read_input(&ifs, str_array, INPUTSIZE);
        ifs.close();
    } else {
        throw std::invalid_argument("Cannot open file");
    }
    std::cout << std::endl;

    /*std::cout << "read input" << std::endl;
    for (int i = 0; i != INPUTSIZE; ++i) {
        std::cout << str_array[i] << std::endl;
    }
    std::cout << "\n";*/

    sort::expand_rotation(str_array, INPUTSIZE, repr_array);
    /*std::cout << "post expansion" << std::endl;
    for (int i = 0; i != EXPANDEDSIZE; ++i) {
        if (!(i % 65)) {
            std::cout << "< " << i / 65 << " >\n";
        }
        std::cout << repr_array[i] << std::endl;
    }*/

    // Scan for distribution
    unsigned int partition_freq[sort::PARTITION_SIZE] = {};
    unsigned int frequency[sort::RADIX_LEVELS][sort::RADIX_SIZE] = {};
    sort::count_frequency(repr_array, EXPANDEDSIZE, partition_freq, frequency);

    std::cerr << "partition frequency:\n";
    std::cerr << "(sum = "
              << std::accumulate(partition_freq,
                                 partition_freq + sort::PARTITION_SIZE, 0)
              << ")\n";
    /*for (int i = 0; i != sort::PARTITION_SIZE; ++i) {
        std::cout << partition_freq[i] << " ";
        if (!(i % 80) && i != 0) {
            std::cout << "\n";
        }
    }
    std::cout << "\n";*/

    // Partition
    std::cout << "check partition" << std::endl;
    sort::partitioning(repr_array, EXPANDEDSIZE, partition_freq);
    std::cout << "post partitioning" << std::endl;
    for (int i = 0; i != EXPANDEDSIZE; ++i) {
        // std::cout << repr_array[i] << std::endl;
    }

    // Sort
    std::cerr << "check sorting\n";
    sort::radix_sort(repr_array, EXPANDEDSIZE, frequency);
    std::cout <<"post sorting" << std::endl;
    for (int i = 0; i != EXPANDEDSIZE; ++i) {
        std::cout << repr_array[i] << std::endl;
    }

    // cleanup
    delete str_array;
    delete repr_array;
}
