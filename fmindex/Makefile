#
# CUDA version compile scripts
# Copyright (C) 2019  Hsuan-Ting Lu
#
# GNU General Public License v3.0+
# (see LICENSE or https://www.gnu.org/licenses/)
#

.PHONY: clean run

# target: GTX1080ti with CUDA-9.0 runtime
CXX=nvcc -std=c++14 -arch=sm_61 -O3
objects = main.o types.o kernel.o gpu_radix_sort.o

all : fmindex

%.o: %.cpp
	$(CXX) -x cu -I. -dc $< -o $@

%.o: %.cu
	$(CXX) -x cu -I. -dc $< -o $@

fmindex : $(objects)
	$(CXX) $(objects) -o fmindex.out

run: fmindex
	./fmindex ../special_1.txt

clean:
	rm *.o program.out
