#include <stdlib.h>
#include <time.h>

#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>

using namespace std;

//-----------------------DO NOT
// CHANGE-------------------------------------------- Read file to get reads
char **inputReads(char *file_path, int *read_count, int *length) {
    FILE *read_file = fopen(file_path, "r");
    int ch, lines = 0;
    char **reads;
    do {
        ch = fgetc(read_file);
        if (ch == '\n') lines++;
    } while (ch != EOF);
    rewind(read_file);
    reads = (char **)malloc(lines * sizeof(char *));
    *read_count = lines;
    int i = 0;
    size_t len = 0;
    for (i = 0; i < lines; i++) {
        reads[i] = NULL;
        len = 0;
        getline(&reads[i], &len, read_file);
    }
    fclose(read_file);
    int j = 0;
    while (reads[0][j] != '\n') j++;
    *length = j + 1;
    for (i = 0; i < lines; i++) reads[i][j] = '$';
    return reads;
}

//-----------------------DO NOT
// CHANGE--------------------------------------------

//-----------------------CHANGE AS
// REQUIRED--------------------------------------------

// Rotate read by 1 character
void rotateRead(char *read, char *rotatedRead, int length) {
    for (int i = 0; i < length - 1; i++) rotatedRead[i] = read[i + 1];
    rotatedRead[length - 1] = read[0];
}

// Generate Sufixes and their SA's for a read
char **generateSuffixes(char *read, int length, int read_id, int **SA) {
    char **suffixes = (char **)malloc(length * sizeof(char *));
    suffixes[0] = (char *)malloc(length * sizeof(char));
    SA[0][0] = 0;
    SA[0][1] = read_id;
    for (int j = 0; j < length; j++) suffixes[0][j] = read[j];
    for (int i = 1; i < length; i++) {
        suffixes[i] = (char *)malloc(length * sizeof(char));
        SA[i][0] = i;
        SA[i][1] = read_id;
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
void makeFMIndex(char ***suffixes, int ***SA, int read_count, int read_length,
                 int F_count[], int **L_count, char *L, int **SA_Final) {
    int i, j;

    // Temporary storage for collecting together all suffixes
    char **temp_suffixes =
        (char **)malloc(read_count * read_length * sizeof(char *));

    // Initalization of temporary storage
    for (i = 0; i < read_count; i++) {
        for (j = 0; j < read_length; j++) {
            temp_suffixes[i * read_length + j] =
                (char *)malloc(read_length * sizeof(char));
            memcpy(&temp_suffixes[i * read_length + j], &suffixes[i][j],
                   read_length * sizeof(char));
            SA_Final[i * read_length + j][0] = SA[i][j][0];
            SA_Final[i * read_length + j][1] = SA[i][j][1];
        }
    }

    char *temp = (char *)malloc(read_length * sizeof(char));
    int temp_int = 0;

    // Focus on improving this for evaluation purpose
    // Sorting of suffixes
    for (i = 0; i < read_count * read_length - 1; i++) {
        for (j = 0; j < read_count * read_length - i - 1; j++) {
            if (compSuffixes(temp_suffixes[j], temp_suffixes[j + 1],
                             read_length) > 0) {
                memcpy(temp, temp_suffixes[j], read_length * sizeof(char));
                memcpy(temp_suffixes[j], temp_suffixes[j + 1],
                       read_length * sizeof(char));
                memcpy(temp_suffixes[j + 1], temp, read_length * sizeof(char));
                temp_int = SA_Final[j][0];
                SA_Final[j][0] = SA_Final[j + 1][0];
                SA_Final[j + 1][0] = temp_int;
                temp_int = SA_Final[j][1];
                SA_Final[j][1] = SA_Final[j + 1][1];
                SA_Final[j + 1][1] = temp_int;
            }
        }
    }

    free(temp);
    char this_F = '$';
    j = 0;

    // Calculation of F_count's
    for (i = 0; i < read_count * read_length; i++) {
        int count = 0;
        while (temp_suffixes[i][0] == this_F) {
            count++;
            i++;
        }
        F_count[j++] = j == 0 ? count : count + 1;
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
}

int main(int argc, char *argv[]) {
    int read_count = 0;   // Total number of reads in file
    int read_length = 0;  // Length of each read in file

    char **reads = inputReads(argv[1], &read_count,
                              &read_length);  // Input reads from file
    char ***suffixes = (char ***)malloc(
        read_count * sizeof(char **));  // Storage for read-wise suffixes
    int ***SA = (int ***)malloc(read_length *
                                sizeof(int **));  // Storage for read-wise SA's

    // Final Values that will be compared for correctness
    // You may change the function prototypes and definitions, but you need to
    // present final results in these arrays
    //-----------------------------Structures for correctness
    // check----------------------------------------------
    int **L_counts =
        (int **)malloc(read_length * read_count *
                       sizeof(int *));  // Final storage for counts of A,C,G and
                                        // T's in last column of sorted suffixes
    char *L = (char *)malloc(
        read_count * read_length *
        sizeof(char *));  // Final storage for last column of sorted suffixes
    int F_counts[] = {0, 0, 0, 0};  // Final counts of consecutive $,A,C and G's
                                    // in firt column of suffixes
    int **SA_Final = (int **)malloc(read_count * read_length * sizeof(int));
    //-----------------------------Structures for correctness
    // check----------------------------------------------

    // Initialisations
    for (int i = 0; i < read_count; i++) {
        SA[i] = (int **)malloc(read_length * sizeof(int *));
        for (int j = 0; j < read_length; j++) {
            SA[i][j] = (int *)malloc(2 * sizeof(int));
            L_counts[i * read_length + j] = (int *)malloc(4 * sizeof(int));
            SA_Final[i * read_length + j] = (int *)malloc(2 * sizeof(int));
            for (int k = 0; k < 4; k++) L_counts[i * read_length + j][k] = 0;
            SA_Final[i * read_length + j][0] = 0;
            SA_Final[i * read_length + j][1] = 0;
        }
    }

    //-----------Time capture start--------------------
    // Generate read-wise suffixes
    for (int i = 0; i < read_count; i++) {
        suffixes[i] = generateSuffixes(reads[i], read_length, i, SA[i]);
    }

    // Calculate finl FM-Index
    makeFMIndex(suffixes, SA, read_count, read_length, F_counts, L_counts, L,
                SA_Final);

    //------------Time capture end----------------------

    // For debug purpose
    for (int i = 0; i < read_count * read_length; i++)
        cout << L[i] << "\t" << SA_Final[i][0] << "," << SA_Final[i][1] << "\t"
             << L_counts[i][0] << "," << L_counts[i][1] << "," << L_counts[i][2]
             << "," << L_counts[i][3] << endl;

    return 0;
}
