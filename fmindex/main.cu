/*
 * Host side master program
 * Copyright (C) 2019  Hsuan-Ting Lu
 *
 * GNU General Public License v3.0+
 * (see LICENSE or https://www.gnu.org/licenses/)
 */

//#include <bitset>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ratio>
#include <vector>
#include <array>

#include "types.hpp"

constexpr const int INPUTSIZE = 10 * std::mega::num;  // Number of reads;
constexpr const int READLENGTH = 64;

// TODO: pin memory
void read_input(std::ifstream* ifs, entry* array) {
    char tmp[64];
    for (int str_idx = 0; str_idx != INPUTSIZE; ++str_idx) {
        ifs->read(tmp, 64);
        array[str_idx] = entry(tmp);
        ifs->ignore();
    }
}

entry data[INPUTSIZE];
int main(int argc, char** argv) {
    // Read to memory
    std::ifstream ifs(argv[1], std::ifstream::in);
    if (ifs) {
        read_input(&ifs, ::data);
        ifs.close();
    }

    // prefix partitioning, save index into multi-index table
    //   sliding window gets all prefixes of rotations (the part needed for
    //   partitioning)
    /* TODO: get GPU mem size
     *
     * Original Data :: char size( =1Byte) * string length( =65) * 10M strings = 650MB
     * Rotated  Data :: 650MB * rotations( =65) = 42.25GB
     * partition size = 
     *
     * GOAL: partition size EQUAL gpu mem size
     */

    // pick one partition, start sorting :: MSD radix sort

    // Print data
    for (char* str : ::data) {
        // custom string print length :: READLENGTH + 1
        printf("%.65s\n", str);
    }
}
