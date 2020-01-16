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
#include <future>

#include "types.hpp"
#include "gpu_radix_sort.hpp"
#include "TA.hpp"
// clang-format on

// TODO: pin memory
void read_input(std::ifstream* ifs, entry* array, char (*TA_array)[64],
                const int INPUTSIZE) {
    char buffer[64];
    buffer[63] = '$';
    for (int str_idx = 0; str_idx != INPUTSIZE; ++str_idx) {
        ifs->read(buffer, 63);
        ifs->ignore();
        std::memcpy(TA_array[str_idx], buffer, 64);
        array[str_idx] = entry(buffer);
    }
}

int main(int argc, char** argv) {
    if (argc != 1 + 1) {
        throw std::invalid_argument("1 arguments needed");
    }

    // Read to memory
    std::ifstream ifs(argv[1], std::ifstream::in);
    const int INPUTSIZE = std::count(std::istreambuf_iterator<char>(ifs),
                                     std::istreambuf_iterator<char>(), '\n');
    ifs.seekg(0);  // rewind
    const int EXPANDEDSIZE = 64 * INPUTSIZE;
    std::cerr << "expected output size :: str_array: " << INPUTSIZE
              << ", rotate_expand: " << EXPANDEDSIZE << "\n";

    // Allocate array
    // TODO: also do cuda-version pinned host malloc
    entry* str_array = new entry[INPUTSIZE];
    entry_repr::origin = str_array;
    entry_repr* repr_array = new entry_repr[EXPANDEDSIZE];
    // allocate TA's array
    char(*TA_str_array)[64] = new char[INPUTSIZE][64];
    char** TA_4b_sorted_suffixes =
        new char*[INPUTSIZE];  // expanded string array
    // TA's structures for correctness check
    char** student_4b_sorted_suffixes = new char*[INPUTSIZE];
    // Init TA's structures
    // TODO: what to do?

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

    if (1) { //std::stoi(argv[2])
        std::cerr << "Measure TA time\n";
        pipeline(TA_str_array, 64, INPUTSIZE, TA_4b_sorted_suffixes);
        mergeAllSorted4bitSuffixes(TA_4b_sorted_suffixes, INPUTSIZE, 64);
    }
    auto TA_timer_end = std::chrono::high_resolution_clock::now();
    double TA_time_spent =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                TA_timer_end - TA_timer_start)
                .count()) /
        1000000;
    std::cout << "TA code spent: " << TA_time_spent << " s" << std::endl;
    delete[] TA_str_array;
    /************************************
     *                                  *
     *   TA's code: TIME CAPTURE ENDS   *
     *                                  *
     ************************************
     */

    auto student_timer_start = std::chrono::high_resolution_clock::now();
    std::cout << "read input" << std::endl;
    /*for (int i = 0; i != INPUTSIZE; ++i) {
        std::cout << str_array[i] << std::endl;
    }
    std::cout << "\n";*/

    sort::expand_rotation(INPUTSIZE, repr_array);
    std::cout << "post expansion" << std::endl;
    /*for (int i = 0; i != EXPANDEDSIZE; ++i) {
        if (!(i % 64)) {
            std::cout << "< " << i / 64 << " >\n";
        }
        std::cout << repr_array[i] << " " << (unsigned
    int)(repr_array[i].str_shift) << std::endl;
    }*/

    // Sort
    std::cerr << "check sorting\n";
    sort::radix_sort(repr_array, INPUTSIZE);
    std::cout << "post sorting" << std::endl;
    /*for (int i = 0; i != EXPANDEDSIZE; ++i) {
        std::cout << repr_array[i] << std::endl;
    }*/

    // FIXME: Fulfill TA's specifications: expand and encode

    char (*result_array)[32];
    sort::encode(entry_array,repr_array, EXPANDEDSIZE,result_array);

    auto student_timer_end = std::chrono::high_resolution_clock::now();
    double student_time_spent =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                student_timer_end - student_timer_start)
                .count()) /
        1000000;
    std::cout << "STUDENT CODE spent: " << student_time_spent << "s" << std::endl;

    // Correctness check and speedup calculation
    if (1) { // std::stoi(argv[2])
        if (checker(INPUTSIZE, TA_4b_sorted_suffixes,
                    student_4b_sorted_suffixes) == 1) {
            std::cout << "answer correct" << std::endl;
        }
        double speedup = TA_time_spent / student_time_spent;
        std::cout << "Speedup=" << speedup << std::endl;
    }

    // cleanup
    delete[] str_array;
    delete[] repr_array;
    delete[] TA_4b_sorted_suffixes;
    delete[] student_4b_sorted_suffixes;
}
