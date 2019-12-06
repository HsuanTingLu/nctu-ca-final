/*
 * Host side master program
 * Copyright (C) 2019  Hsuan-Ting Lu
 *
 * GNU General Public License v3.0+
 * (see LICENSE or https://www.gnu.org/licenses/)
 */

//#include <bitset>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <ratio>

constexpr const int INPUTSIZE = 1 * std::mega::num;  // Number of reads;
constexpr const int GPU_MEM = 11 * 1024 * 1048576;   // Bytes
constexpr const int READLENGTH = 64;

// TODO: pin memory
__host__
inline void read_input(std::ifstream* ifs, char (*data)[READLENGTH+1]) {
    for (int str_idx = 0; str_idx != INPUTSIZE; ++str_idx) {
        ifs->read(data[str_idx], READLENGTH);
        data[str_idx][READLENGTH] = '$';
        ifs->ignore();
    }
}

__global__
char data[INPUTSIZE][READLENGTH+1];
int main(int argc, char** argv) {
    // Read to memory
    std::ifstream ifs(argv[1], std::ifstream::in);
    if (ifs) {
        read_input(&ifs, data);
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
    for (char* str : data) {
        // custom string print length :: READLENGTH + 1
        printf("%.65s\n", str);
    }
}
