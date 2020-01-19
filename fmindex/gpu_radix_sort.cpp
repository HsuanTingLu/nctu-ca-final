/*
 * CPU-only multi-thread radix sort
 * Copyright (C) 2019  Hsuan-Ting Lu
 *
 * GNU General Public License v3.0+
 * (see LICENSE or https://www.gnu.org/licenses/)
 */

// clang-format off
#include <cstdlib>
#include <cstring>

#include <memory>
#include <future>
#include <mutex>
#include <iostream>  // DEBUG:

#include "gpu_radix_sort.hpp"
#include "kernels.cu"
// clang-format on
// DEBUG:
#define RED(x) "\033[31m" x "\033[0m"
#define GREEN(x) "\033[32m" x "\033[0m"
#define YELLOW(x) "\033[33m" x "\033[0m"

namespace sort {

void expand_rotation(const int array_size, entry_repr *repr_array) {
    /* Expands and creates the entire table of representations of
     * strings-to-be-sorted
     *
     * NOTE: cannot use async-task-launch because TA's machines sucks,
     *  cannot support that much thread per process
     */
    for (int str_idx = 0; str_idx != array_size; ++str_idx) {
        // Splitted one loop into two in case of needing parallelisation
        int repr_counter = str_idx * 64;
        for (int str_shift = 0; str_shift != 64; ++str_shift) {
            repr_array[repr_counter].str_idx = str_idx;
            repr_array[repr_counter].str_shift = str_shift;
            ++repr_counter;
        }
    }
}  // namespace sort

void radix_sort(entry_repr *repr_array, const unsigned int entry_array_size) {
    const unsigned int repr_array_size = entry_array_size * 64;
    // Set CUDA kernel launch configurations
    constexpr const int threadsPerBlock = 1024;
    const int blocksPerGrid =
        (repr_array_size + threadsPerBlock - 1) / threadsPerBlock;

    // GPU working area
    entry *gpu_entry_array;
    entry_repr *gpu_repr_array;
    entry_repr *gpu_alt_array;
    cudaMalloc(&gpu_entry_array, entry_array_size * sizeof(entry));
    cudaMalloc(&gpu_repr_array, repr_array_size * sizeof(entry_repr));
    cudaMalloc(&gpu_alt_array, repr_array_size * sizeof(entry_repr));

    // Init GPU working area
    cudaMemcpy(gpu_entry_array, entry_repr::origin,
               entry_array_size * sizeof(entry), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_repr_array, repr_array, repr_array_size * sizeof(entry_repr),
               cudaMemcpyHostToDevice);

    // alternation pointers
    entry_repr *gpu_from = gpu_repr_array, *gpu_to = gpu_alt_array;

    // allocate tmp workspaces on device
    unsigned int *gpu_bucket_indexes;
    unsigned int *gpu_bucket_key_label;
    unsigned int *gpu_bucket_HEADs;
    cudaMalloc(&gpu_bucket_indexes, repr_array_size * sizeof(unsigned int));
    cudaMalloc(&gpu_bucket_key_label, repr_array_size * sizeof(unsigned int));
    cudaMalloc(&gpu_bucket_HEADs, sort::RADIX_SIZE * sizeof(unsigned int));

    // re-use storage across passes
    unsigned int *bucket_indexes;
    unsigned int *bucket_key_label;
    unsigned int *bucket_HEADs;
    unsigned int *frequency;
    cudaHostAlloc(&bucket_indexes, repr_array_size * sizeof(unsigned int),
                  cudaHostAllocDefault);
    cudaHostAlloc(&bucket_key_label, repr_array_size * sizeof(unsigned int),
                  cudaHostAllocDefault);
    cudaHostAlloc(&bucket_HEADs, sort::RADIX_SIZE * sizeof(unsigned int),
                  cudaHostAllocDefault);
    cudaHostAlloc(&frequency, sort::RADIX_SIZE * sizeof(unsigned int),
                  cudaHostAllocDefault);

    for (unsigned int pass = 0; pass != RADIX_LEVELS; ++pass) {
        std::cout << "pass: " << pass << std::endl;
        // init arrays
        for (unsigned int i = 0; i != sort::RADIX_SIZE; ++i) {
            frequency[i] = 0U;
        }

        // TODO: set constants in device constant memory
        // pass, array_size

        // calculate bucket indexes
        calc_bucket_index<<<blocksPerGrid, threadsPerBlock>>>(
            pass, gpu_entry_array, gpu_from, repr_array_size,
            gpu_bucket_indexes);
        // TODO: use async memory op
        cudaMemcpy(bucket_indexes, gpu_bucket_indexes,
                   repr_array_size * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);

        // FIXME: DEBUG:
        /*for(unsigned int i=0; i!=repr_array_size; ++i) {
            std::cout << bucket_indexes[i] << std::endl;
        }*/

        // create data histogram
        for (unsigned int repr_idx = 0; repr_idx != repr_array_size;
             ++repr_idx) {
            unsigned int bucket_idx = bucket_indexes[repr_idx];
            bucket_key_label[repr_idx] = frequency[bucket_idx];
            frequency[bucket_idx] += 1;
        }
        cudaMemcpy(gpu_bucket_key_label, bucket_key_label,
                   repr_array_size * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
        // Init bucket HEADs (bucket HEAD pointers)
        unsigned int next = 0;
        for (unsigned int bucket_idx = 0; bucket_idx != sort::RADIX_SIZE;
             ++bucket_idx) {
            bucket_HEADs[bucket_idx] = next;
            next += frequency[bucket_idx];
        }
        cudaMemcpy(gpu_bucket_HEADs, bucket_HEADs,
                   sort::RADIX_SIZE * sizeof(unsigned int),
                   cudaMemcpyHostToDevice);
        cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 102400);
        // actually move data to buckets
        move_to_buckets<<<blocksPerGrid, threadsPerBlock>>>(
            gpu_from, gpu_to, repr_array_size, gpu_bucket_HEADs,
            gpu_bucket_key_label, gpu_bucket_indexes);

        // FIXME:
        entry_repr *debugg = new entry_repr[repr_array_size];
        cudaMemcpy(debugg, gpu_to, repr_array_size * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        /*for (unsigned int i = 0; i != repr_array_size; ++i) {
          std::cout << debugg[i] << std::endl;
        }*/

        // swap arrays (via pointers {from}/{to} swapping)
        entry_repr *gpu_ptr_swap_tmp = gpu_from;
        gpu_from = gpu_to;
        gpu_to = gpu_ptr_swap_tmp;
    }

    // return the correct array if ${RADIX_LEVELS} is odd
    cudaMemcpy(repr_array, gpu_repr_array, repr_array_size * sizeof(entry_repr),
               cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(gpu_bucket_indexes);
    cudaFree(gpu_bucket_key_label);
    cudaFree(gpu_bucket_HEADs);

    cudaFree(gpu_entry_array);
    cudaFree(gpu_to);
    cudaFree(gpu_from);

    cudaFreeHost(bucket_indexes);
    cudaFreeHost(bucket_key_label);
    cudaFreeHost(bucket_HEADs);
    cudaFreeHost(frequency);
}

void encode(entry *entry_array, entry_repr *repr_array,
            unsigned int repr_array_size, char (*result_array)[32]) {
    constexpr const int threadsPerBlock = 1024;
    const int blocksPerGrid =
        (repr_array_size + threadsPerBlock - 1) / threadsPerBlock;

    expand_and_encode<<<blocksPerGrid, threadsPerBlock>>>(
        entry_array, repr_array, repr_array_size, result_array);
}

}  // namespace sort
