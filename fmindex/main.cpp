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
#include <chrono>
#include <numeric>
#include <algorithm>

#include "types.hpp"
#include "parallel_radix_sort.hpp"
#include "TA.hpp"
// clang-format on

// TODO: pin memory
void read_input(std::ifstream* ifs, entry* array, char (*TA_array)[65],
                const int INPUTSIZE) {
    char buffer[65];
    buffer[64] = '$';
    for (int str_idx = 0; str_idx != INPUTSIZE; ++str_idx) {
        ifs->read(buffer, 64);
        ifs->ignore();
        std::memcpy(TA_array[str_idx], buffer, 65);
        array[str_idx] = entry(buffer);
    }
}

int main(int argc, char** argv) {
    if (argc != 1 + 2) {
        throw std::invalid_argument("2 arguments needed");
    }

    // Read to memory
    std::ifstream ifs(argv[1], std::ifstream::in);
    const int INPUTSIZE = std::count(std::istreambuf_iterator<char>(ifs),
                                     std::istreambuf_iterator<char>(), '\n');
    ifs.seekg(0);  // rewind
    const int EXPANDEDSIZE = 65 * INPUTSIZE;
    std::cerr << "expected output size :: str_array: " << INPUTSIZE
              << ", rotate_expand: " << EXPANDEDSIZE << "\n";

    // Allocate array
    entry* str_array = new entry[INPUTSIZE];
    entry_repr::origin = str_array;
    entry_repr* repr_array = new entry_repr[EXPANDEDSIZE];
    // allocate TA's array
    auto TA_str_array = new char[INPUTSIZE][65];
    auto TA_suffixes = new char**[INPUTSIZE];  // expanded string array
    auto TA_L = new char[EXPANDEDSIZE];
    auto TA_F_counts = new int[4]{0, 0, 0, 0};
    int** TA_SA_Final = nullptr;
    int** TA_L_counts = nullptr;
    // TA's structures for correctness check
    int** student_SA_Final = nullptr;
    int** student_L_counts = nullptr;
    char* student_L = nullptr;
    int student_F_counts[4] = {0, 0, 0, 0};

    // Read input
    read_input(&ifs, str_array, TA_str_array, INPUTSIZE);
    ifs.close();

    std::cout << std::endl;

    /************************************
     *                                  *
     *  TA's code: TIME CAPTURE STARTS  *
     *                                  *
     ************************************
     */
    auto TA_timer_start = std::chrono::high_resolution_clock::now();

    if (std::stoi(argv[2])) {
        // FIXME: TA starts FM-index generation
        for (int i = 0; i != INPUTSIZE; ++i) {
            TA_suffixes[i] = generateSuffixes(TA_str_array[i], 65);
        }
        TA_L_counts = makeFMIndex(TA_suffixes, INPUTSIZE, 65, TA_F_counts, TA_L,
                                  TA_SA_Final);
    }
    auto TA_timer_end = std::chrono::high_resolution_clock::now();
    delete[] TA_str_array;
    delete[] TA_suffixes;
    /************************************
     *                                  *
     *   TA's code: TIME CAPTURE ENDS   *
     *                                  *
     ************************************
     */

    auto student_timer_start = std::chrono::high_resolution_clock::now();
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

    std::cerr << "counting frequency\n";
    sort::count_frequency(repr_array, EXPANDEDSIZE, partition_freq, frequency);
    std::cerr << "post frequency counting\n";

    // partition frequency sanity check
    std::cerr << "partition frequency:\n";
    int partition_frequency_sum = std::accumulate(
        partition_freq, partition_freq + sort::PARTITION_SIZE, 0);
    std::cerr << "(sum = " << partition_frequency_sum << ")\n";
    // sorting frequency sanity check
    std::cerr << "sort frequency:\n";
    for (unsigned int level = 0; level != sort::RADIX_LEVELS; ++level) {
        int sort_frequency_sum = std::accumulate(
            frequency[level], frequency[level] + sort::RADIX_SIZE, 0);
        std::cerr << "level " << level << ": sum = " << sort_frequency_sum
                  << "\n";
        if (sort_frequency_sum != EXPANDEDSIZE) {
            throw std::logic_error("sorting pass frequency sum does not match");
        }
    }

    /*for (int i = 0; i != sort::PARTITION_SIZE; ++i) {
        std::cout << partition_freq[i] << " ";
        if (!(i % 80) && i != 0) {
            std::cout << "\n";
        }
    }
    std::cout << "\n";*/

    // Partition
    std::cout << "do partition" << std::endl;
    sort::partitioning(repr_array, EXPANDEDSIZE, partition_freq);
    std::cout << "post partitioning" << std::endl;
    /*for (int i = 0; i != EXPANDEDSIZE; ++i) {
        std::cout << repr_array[i] << std::endl;
    }*/

    // Sort
    std::cerr << "check sorting\n";
    sort::radix_sort(repr_array, EXPANDEDSIZE, frequency);
    std::cout << "post sorting" << std::endl;
    for (int i = 0; i != EXPANDEDSIZE; ++i) {
        std::cout << repr_array[i] << std::endl;
    }

    auto student_timer_end = std::chrono::high_resolution_clock::now();
    double student_time_spent =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                student_timer_end - student_timer_start)
                .count()) /
        1000000;
    std::cout << "spent: " << student_time_spent << "s" << std::endl;

    // Correctness check and speedup calculation
    if (std::stoi(argv[2])) {
        double TA_time_spent =
            static_cast<double>(
                std::chrono::duration_cast<std::chrono::microseconds>(
                    TA_timer_end - TA_timer_start)
                    .count()) /
            1000000;
        std::cout << "TA code spent: " << TA_time_spent << " s" << std::endl;

        double speedup = 0.0;
        if (checker(INPUTSIZE, 65, student_L, student_SA_Final,
                    student_L_counts, student_F_counts, TA_L, TA_SA_Final,
                    TA_L_counts, TA_F_counts) == 1) {
            speedup = TA_time_spent / student_time_spent;
        }
        std::cout << "Speedup=" << speedup << std::endl;
    }

    // cleanup
    delete[] str_array;
    delete[] repr_array;
}
