#ifndef TA_HPP_
#define TA_HPP_

// clang-format off
#include <cstdlib>

#include <iostream>
// clang-format on

// Check correctness of values
int checker(int read_count, char **fourbit_sorted_suffixes_original,
            char **fourbit_sorted_suffixes_student) {
    int correct = 1;
    for (int i = 0; i < read_count * 64; i++) {
        for (int j = 0; j < 64 / 2; j++) {
            if (fourbit_sorted_suffixes_student[i][j] !=
                fourbit_sorted_suffixes_original[i][j])
                correct = 0;
        }
    }
    return correct;
}

// Rotate 4-bit encoded read by 1 character (4-bit)
char *rotateRead(char *read, int byte_length) {
    char prev_4bit = (read[0] & 0x0F) << 4;
    read[0] = (read[0] >> 4) & 0x0F;
    for (int i = 1; i < byte_length; i++) {
        char this_char = ((read[i] >> 4) & 0x0F) | prev_4bit;
        prev_4bit = (read[i] & 0x0F) << 4;
        read[i] = this_char;
    }
    read[0] = read[0] | prev_4bit;
    char *rotated_read = (char *)malloc(byte_length * sizeof(char));
    for (int i = 0; i < byte_length; i++) rotated_read[i] = read[i];
    return rotated_read;
}

// Generate Sufixes for a 4-bit encoded read
char **generateSuffixes(char *read, int byte_length) {
    char **suffixes = (char **)malloc(byte_length * 2 * sizeof(char *));
    for (int i = 0; i < byte_length * 2; i++) {
        suffixes[i] = rotateRead(read, byte_length);
    }
    return suffixes;
}

// Comparator for 4-bit encoded Suffixes
int compSuffixes(char *suffix1, char *suffix2, int byte_length) {
    int ret = 0;
    for (int i = 0; i < byte_length; i++) {
        if (suffix1[i] > suffix2[i])
            return 1;
        else if (suffix1[i] < suffix2[i])
            return -1;
    }
    return ret;
}

char *fourbitEncodeRead(char *read, int length) {
    int byte_length = length / 2;
    char *fourbit_read = (char *)calloc(byte_length, sizeof(char));
    for (int i = 0; i < length; i++) {
        char this_char = read[i];
        char fourbit_char;
        if (this_char == '$')
            fourbit_char = 0x00;
        else if (this_char == 'A')
            fourbit_char = 0x01;
        else if (this_char == 'C')
            fourbit_char = 0x02;
        else if (this_char == 'G')
            fourbit_char = 0x03;
        else
            fourbit_char = 0x04;
        fourbit_char = i % 2 == 0 ? fourbit_char << 4 : fourbit_char;
        fourbit_read[i / 2] = fourbit_read[i / 2] | fourbit_char;
    }
    return fourbit_read;
}

void sort_fourbit_suffixes(char **suffixes, int suffix_count, int byte_length) {
    char *temp = (char *)malloc(byte_length * sizeof(char));
    for (int i = 0; i < suffix_count - 1; i++) {
        for (int j = 0; j < suffix_count - i - 1; j++) {
            if (compSuffixes(suffixes[j], suffixes[j + 1], byte_length) > 0) {
                memcpy(temp, suffixes[j], byte_length * sizeof(char));
                memcpy(suffixes[j], suffixes[j + 1],
                       byte_length * sizeof(char));
                memcpy(suffixes[j + 1], temp, byte_length * sizeof(char));
            }
        }
    }
}

// Default Pipeline. You need to implement CUDA function corresponding to
// everything inside this function
void pipeline(char (*reads)[64], int read_length, int read_count,
              char **fourbit_sorted_suffixes_original) {
    fourbit_sorted_suffixes_original =
        (char **)malloc(read_length * read_count * sizeof(char *));
    for (int i = 0; i < read_count; i++) {
        char **suffixes_for_read = generateSuffixes(
            fourbitEncodeRead(reads[i], read_length), read_length / 2);
        sort_fourbit_suffixes(suffixes_for_read, read_length, read_length / 2);
        for (int j = 0; j < read_length; j++) {
            fourbit_sorted_suffixes_original[i * read_length + j] =
                suffixes_for_read[j];
        }
    }
    //--------------For debug purpose--------------
    /*
    for(int i=0;i<read_count*read_length;i++){
        for(int j=0;j<read_length/2;j++)
            printf("%x\t",fourbit_sorted_suffixes_original[i][j]);
        printf("\n");
    }*/
    //---------------------------------------------
}

// Merge all sorted suffixes in overall sorted order
void mergeAllSorted4bitSuffixes(char **suffixes, int read_count,
                                int read_length) {}

#endif
