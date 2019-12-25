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
    auto student_SA_Final = new int[EXPANDEDSIZE][2];
    auto student_L_counts = new int[EXPANDEDSIZE][4];
    char* student_L = new char[EXPANDEDSIZE];
    int student_F_counts[4] = {0, 0, 0, 0};
    // Init TA's structures
    for(int i=0; i!=EXPANDEDSIZE; ++i) {
        auto SA = student_SA_Final[i];
        SA[0] = 0;
        SA[1] = 0;
        auto L_count = student_L_counts[i];
        L_count[0] = 0;
        L_count[1] = 0;
        L_count[2] = 0;
        L_count[3] = 0;
    }

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
        std::cerr << "Measure TA time\n";
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
    std::cout << "read input" << std::endl;
    /*for (int i = 0; i != INPUTSIZE; ++i) {
        std::cout << str_array[i] << std::endl;
    }
    std::cout << "\n";*/

    sort::expand_rotation(INPUTSIZE, repr_array);
    std::cout << "post expansion" << std::endl;
    /*for (int i = 0; i != EXPANDEDSIZE; ++i) {
        if (!(i % 65)) {
            std::cout << "< " << i / 65 << " >\n";
        }
        std::cout << repr_array[i] << " " << (unsigned
    int)(repr_array[i].str_shift) << std::endl;
    }*/

    // Scan for distribution
    unsigned int partition_freq[sort::PARTITION_SIZE] = {0};

    // Partition
    std::cout << "do partition" << std::endl;
    sort::partitioning(repr_array, EXPANDEDSIZE, partition_freq);
    std::cout << "post partitioning" << std::endl;
    /*for (int i = 0; i != EXPANDEDSIZE; ++i) {
        std::cout << repr_array[i] << " " << (unsigned
    int)(repr_array[i].str_shift) << std::endl;
    }*/

    // Sort
    std::cerr << "check sorting\n";
    std::future<void> sort_work[sort::PARTITION_SIZE];
    for (unsigned int part = 0; part != sort::PARTITION_SIZE; ++part) {
        entry_repr* subarray_head =
            repr_array +
            std::accumulate(partition_freq, partition_freq + part, 0);
        unsigned int subarray_size = partition_freq[part];
        std::cout << "Start sorting sub-section " << part
                  << ", size = " << subarray_size << std::endl;

        sort_work[part] = std::async(
            std::launch::async, [subarray_head, subarray_size]() -> void {
                sort::radix_sort(subarray_head, subarray_size);
            });
    }
    for (unsigned int part = 0; part != sort::PARTITION_SIZE; ++part) {
        sort_work[part].wait();
        std::cout << "Finish sorting sub-section " << part << std::endl;
    }

    std::cout << "post sorting" << std::endl;
    /*for (int i = 0; i != EXPANDEDSIZE; ++i) {
        std::cout << repr_array[i] << std::endl;
    }*/

    // FIXME: Fulfill TA's specifications
    for(int i=0; i!=4; ++i) {
        student_F_counts[i] = partition_freq[i];
    }
    for(int i=0; i!=EXPANDEDSIZE; ++i) {
        entry_repr repr = repr_array[i];
        uint8_t* string = (repr.origin[repr.str_idx]).data;
        uint8_t L = string[(repr.str_shift+64)%65];
        student_L[i] = utils::reverse_char(L);
        student_SA_Final[i][0] = repr.str_shift;
        student_SA_Final[i][1] = repr.str_idx;

        if (i != 0) {
            student_L_counts[i][0] = student_L_counts[i-1][0];
            student_L_counts[i][1] = student_L_counts[i-1][1];
            student_L_counts[i][2] = student_L_counts[i-1][2];
            student_L_counts[i][3] = student_L_counts[i-1][3];
        }
        if (static_cast<unsigned int>(L) != 0) {
            student_L_counts[i][static_cast<unsigned int>(L)-1] += 1;
        }
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
        //speedup = TA_time_spent / student_time_spent; // DEBUG:
        std::cout << "Speedup=" << speedup << std::endl;
    }

    // cleanup
    delete[] str_array;
    delete[] repr_array;
}
