#ifndef TA_HPP_
#define TA_HPP_

// clang-format off
#include <cstdlib>

#include <iostream>
// clang-format on

// Check correctness of values
int SA_type[1][2];
int Lcnt_type[1][4];
int checker(int read_count, int read_length, char *L_student,
            decltype(SA_type) SA_Final_student,
            decltype(Lcnt_type) L_counts_student, int *F_counts_student,
            char *L, int **SA_Final, int **L_counts, int *F_counts) {
    int correct = 1;
    for (int i = 0; i < read_count * read_length; i++) {
        if (L_student[i] != L[i]) {
            correct = 0;
            std::cerr << "L wrong, ans=" << L[i] << ", given=" << L_student[i]
                      << "\n";
        }
        for (int j = 0; j < 2; j++) {
            if (SA_Final_student[i][j] != SA_Final[i][j]) {
                correct = 0;
                std::cerr << "SA wrong, ans=" << SA_Final[i][j] << ", given=" << SA_Final_student[i][j] << "\n";
            }
        }
        for (int j = 0; j < 4; j++) {
            if (L_counts_student[i][j] != L_counts[i][j]) {
                correct = 0;
                std::cerr << "L_count wrong, " << i << ", " << j << ", ans=" << L_counts[i][j] << ", given=" << L_counts_student[i][j] << "\n";
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        if (F_counts_student[i] != F_counts[i]) {
            correct = 0;
            std::cerr << "F wrong, ans=" << F_counts[i]
                      << ", given=" << F_counts_student[i] << "\n";
        }
    }
    return correct;
}

// Rotate read by 1 character
void rotateRead(char *read, char *rotatedRead, int length) {
    for (int i = 0; i < length - 1; i++) rotatedRead[i] = read[i + 1];
    rotatedRead[length - 1] = read[0];
}

// Generate Sufixes and their SA's for a read
char **generateSuffixes(char *read, int length) {
    char **suffixes = static_cast<char **>(malloc(length * sizeof(char *)));
    suffixes[0] = static_cast<char *>(malloc(length * sizeof(char)));
    for (int j = 0; j < length; j++) {
        suffixes[0][j] = read[j];
    }
    for (int i = 1; i < length; i++) {
        suffixes[i] = static_cast<char *>(malloc(length * sizeof(char)));
        rotateRead(suffixes[i - 1], suffixes[i], length);
    }
    return suffixes;
}

// Comparator for Suffixes
int compSuffixes(char *suffix1, char *suffix2, int length) {
    int ret = 0;
    for (int i = 0; i < length; i++) {
        if (suffix1[i] > suffix2[i])
            return 1;
        else if (suffix1[i] < suffix2[i])
            return -1;
    }
    return ret;
}

// Calculates the final FM-Index
int **makeFMIndex(char ***suffixes, int read_count, int read_length,
                  int F_count[], char *L, int **&SA_Final) {
    int i, j;

    SA_Final =
        static_cast<int **>(malloc(read_count * read_length * sizeof(int *)));
    for (i = 0; i < read_count * read_length; i++) {
        SA_Final[i] = static_cast<int *>(malloc(2 * sizeof(int)));
    }

    // Temporary storage for collecting together all suffixes
    char **temp_suffixes =
        static_cast<char **>(malloc(read_count * read_length * sizeof(char *)));

    // Initalization of temporary storage
    for (i = 0; i < read_count; i++) {
        for (j = 0; j < read_length; j++) {
            temp_suffixes[i * read_length + j] =
                static_cast<char *>(malloc(read_length * sizeof(char)));
            std::memcpy(&temp_suffixes[i * read_length + j], &suffixes[i][j],
                        read_length * sizeof(char));
            SA_Final[i * read_length + j][0] = j;
            SA_Final[i * read_length + j][1] = i;
        }
    }

    char *temp = static_cast<char *>(malloc(read_length * sizeof(char)));

    int **L_count =
        static_cast<int **>(malloc(read_length * read_count * sizeof(int *)));
    for (i = 0; i < read_length * read_count; i++) {
        L_count[i] = static_cast<int *>(malloc(4 * sizeof(int)));
        for (j = 0; j < 4; j++) {
            L_count[i][j] = 0;
        }
    }

    // Focus on improving this for evaluation purpose
    // Sorting of suffixes
    for (i = 0; i < read_count * read_length - 1; i++) {
        for (j = 0; j < read_count * read_length - i - 1; j++) {
            if (compSuffixes(temp_suffixes[j], temp_suffixes[j + 1],
                             read_length) > 0) {
                std::memcpy(temp, temp_suffixes[j], read_length * sizeof(char));
                std::memcpy(temp_suffixes[j], temp_suffixes[j + 1],
                            read_length * sizeof(char));
                std::memcpy(temp_suffixes[j + 1], temp,
                            read_length * sizeof(char));
                int temp_int = SA_Final[j][0];
                SA_Final[j][0] = SA_Final[j + 1][0];
                SA_Final[j + 1][0] = temp_int;
                temp_int = SA_Final[j][1];
                SA_Final[j][1] = SA_Final[j + 1][1];
                SA_Final[j + 1][1] = temp_int;
            }
        }
    }

    delete[] temp;
    char this_F = '$';
    j = 0;

    // Calculation of F_count's
    for (i = 0; i < read_count * read_length; i++) {
        int count = 0;
        while (temp_suffixes[i][0] == this_F) {
            count++;
            i++;
        }
        if (j) {
            F_count[j] = count + 1;
        } else {
            F_count[j] = count;
        }
        j += 1;

        this_F = temp_suffixes[i][0];
        if (temp_suffixes[i][0] == 'T') break;
    }

    // Calculation of L's and L_count's
    for (i = 0; i < read_count * read_length; i++) {
        char ch = temp_suffixes[i][read_length - 1];
        L[i] = ch;
        if (i > 0) {
            for (int k = 0; k < 4; k++) L_count[i][k] = L_count[i - 1][k];
        }
        if (ch == 'A')
            L_count[i][0]++;
        else if (ch == 'C')
            L_count[i][1]++;
        else if (ch == 'G')
            L_count[i][2]++;
        else if (ch == 'T')
            L_count[i][3]++;
    }

    return L_count;
}

#endif