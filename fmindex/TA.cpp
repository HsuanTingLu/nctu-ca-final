#include <sys/time.h>

#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>

// ----DO NOT CHANGE NAMES, ONLY MODIFY VALUES----

// Final Values that will be compared for correctness
// You may change the function prototypes and definitions, but you need to
// present final results in these arrays

// ----Structures for correctness check----
int **SA_Final_student;
int **L_counts_student;
char *L_student;
int F_counts_student[] = {0, 0, 0, 0};
// --------

// --------

//----DO NOT CHANGE----

int read_count = 0;
int read_length = 0;

int **SA_Final;
int **L_counts;
char *L;
int F_counts[] = {0, 0, 0, 0};

// Read file to get reads
char **inputReads(char *file_path, int &read_count, int &length) {
    FILE *read_file = std::fopen(file_path, "r");
    int ch = 0;
    read_count = 0;
    do {
        ch = std::fgetc(read_file);
        if (ch == '\n') read_count++;
    } while (ch != EOF);
    std::rewind(read_file);
    char **reads = new char *[read_count];
    int i = 0;
    size_t len = 0;
    for (i = 0; i < read_count; i++) {
        reads[i] = NULL;
        len = 0;
        getline(&reads[i], &len, read_file);
    }
    std::fclose(read_file);
    int j = 0;
    while (reads[0][j] != '\n') j++;
    length = j + 1;
    for (i = 0; i < read_count; i++) reads[i][j] = '$';
    return reads;
}

// Check correctness of values
int checker() {
    int correct = 1;
    for (int i = 0; i < read_count * read_length; i++) {
        if (L_student[i] != L[i]) correct = 0;
        for (int j = 0; j < 2; j++) {
            if (SA_Final_student[i][j] != SA_Final[i][j]) correct = 0;
        }
        for (int j = 0; j < 4; j++) {
            if (L_counts_student[i][j] != L_counts[i][j]) correct = 0;
        }
    }
    for (int i = 0; i < 4; i++) {
        if (F_counts_student[i] != F_counts[i]) correct = 0;
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
    for (int j = 0; j < length; j++) suffixes[0][j] = read[j];
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
                  int F_count[], char *L) {
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

// ----DO NOT CHANGE----

int main(int argc, char *argv[]) {
    char **reads = inputReads(argv[1], read_count,
                              read_length);  // Input reads from file
    char ***suffixes =
        new char **[read_count];  // Storage for read-wise suffixes

    // ----Structures for correctness check----
    L = new char[read_count * read_length];  // Final storage for last column of
                                             // sorted suffixes
    // ----Structures for correctness check----

    // ----Default implementation----
    // ----Time capture start----
    auto TA_timer_start = std::chrono::high_resolution_clock::now();
    // Generate read-wise suffixes
    for (int i = 0; i < read_count; i++) {
        suffixes[i] = generateSuffixes(reads[i], read_length);
    }

    // Calculate final FM-Index
    L_counts = makeFMIndex(suffixes, read_count, read_length, F_counts, L);

    auto TA_timer_end = std::chrono::high_resolution_clock::now();
    double TA_time_spent =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                TA_timer_end - TA_timer_start)
                .count()) /
        1000000;
    // ----Time capture end----
    //--------

    // ----Your implementations----
    auto student_timer_start = std::chrono::high_resolution_clock::now();
    // ----Call your functions here----

    // ----Call your functions here----
    auto student_timer_end = std::chrono::high_resolution_clock::now();
    double student_time_spent =
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(
                student_timer_end - student_timer_start)
                .count()) /
        1000000;
    // --------

    // ----For debug purpose only----
    /*for (int i = 0; i < read_count * read_length; i++)
        std::cout << L[i] << "\t" << SA_Final[i][0] << "," << SA_Final[i][1]
                  << "\t" << L_counts[i][0] << "," << L_counts[i][1] << ","
                  << L_counts[i][2] << "," << L_counts[i][3] << std::endl;
    */
    // --------

    // ----Correction check and speedup calculation----
    double speedup = 0.0;
    std::cout << "spent time " << TA_time_spent << "\n";
    /*if (checker() == 1) {
        speedup = TA_time_spent / student_time_spent;
    }*/
    std::cout << "Speedup=" << speedup << std::endl;
    // --------
    return 0;
}
