#include <cstring>
#include <cstdio>

#include <array>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>

constexpr const int READLENGTH = 64;

inline void read_input(std::ifstream& ifs,
                       std::vector<std::array<char, READLENGTH + 1>>& data,
                       const int INPUTSIZE) {
    for (int str_idx = 0; str_idx != INPUTSIZE; ++str_idx) {
        std::array<char, READLENGTH + 1> tmp;
        ifs.read(tmp.data(), READLENGTH);
        tmp[READLENGTH] = '$';
        data.push_back(tmp);
        ifs.ignore();
    }
}

int main(int argc, char** argv) {
    if (argc != 1 + 2) {  // DEBUG:
        throw std::invalid_argument("2 arguments needed");
    }
    // HACK: use "wc -l" for file line counting
    const int INPUTSIZE = std::stoi(argv[2]);
    std::cout << "expected str_array size: " << INPUTSIZE << std::endl;

    std::vector<std::array<char, READLENGTH + 1>> data;
    data.reserve(INPUTSIZE);

    std::ifstream ifs(argv[1], std::ifstream::in);
    if (ifs) {
        read_input(ifs, data, INPUTSIZE);
        ifs.close();
    }

    std::sort(data.begin(), data.end(),
              [](std::array<char, READLENGTH + 1>& a,
                 std::array<char, READLENGTH + 1>& b) {
                  for (int i = 0; i != READLENGTH + 1; ++i) {
                      if (a[i] < b[i]) {
                          return true;
                      } else if (a[i] > b[i]) {
                          return false;
                      }
                  }
                  return false;
              });

    //for (auto str : data) {
        //std::printf("%.65s\n", str.data());
    //}
}