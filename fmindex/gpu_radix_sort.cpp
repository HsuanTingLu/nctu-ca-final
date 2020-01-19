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

void radix_sort(entry *gpu_entry_array, entry_repr *gpu_repr_array,
                entry_repr *gpu_alt_array,
                const unsigned int entry_array_size) {
    const unsigned int repr_array_size = entry_array_size * 64;
    // Set CUDA kernel launch configurations
    constexpr const int threadsPerBlock = 1024;
    const int blocksPerGrid =
        (repr_array_size + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t stream_bkt_idx, stream_bkt_HEAD, stream_key_lbl;
    cudaStreamCreate(&stream_bkt_idx);
    cudaStreamCreate(&stream_bkt_HEAD);
    cudaStreamCreate(&stream_key_lbl);

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
    cudaMallocHost(&bucket_indexes, repr_array_size * sizeof(unsigned int));
    cudaMallocHost(&bucket_key_label, repr_array_size * sizeof(unsigned int));
    cudaMallocHost(&bucket_HEADs, sort::RADIX_SIZE * sizeof(unsigned int));
    cudaMallocHost(&frequency, sort::RADIX_SIZE * sizeof(unsigned int));

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
        cudaMemcpyAsync(bucket_indexes, gpu_bucket_indexes,
                        repr_array_size * sizeof(unsigned int),
                        cudaMemcpyDeviceToHost, stream_bkt_idx);
        cudaStreamSynchronize(stream_bkt_idx);

        // create data histogram
        for (unsigned int repr_idx = 0; repr_idx != repr_array_size;
             ++repr_idx) {
            unsigned int bucket_idx = bucket_indexes[repr_idx];
            bucket_key_label[repr_idx] = frequency[bucket_idx];
            frequency[bucket_idx] += 1;
        }
        cudaMemcpyAsync(gpu_bucket_key_label, bucket_key_label,
                        repr_array_size * sizeof(unsigned int),
                        cudaMemcpyHostToDevice, stream_key_lbl);
        cudaStreamSynchronize(stream_key_lbl);

        // Init bucket HEADs (bucket HEAD pointers)
        unsigned int next = 0;
        for (unsigned int bucket_idx = 0; bucket_idx != sort::RADIX_SIZE;
             ++bucket_idx) {
            bucket_HEADs[bucket_idx] = next;
            next += frequency[bucket_idx];
        }
        cudaMemcpyAsync(gpu_bucket_HEADs, bucket_HEADs,
                        sort::RADIX_SIZE * sizeof(unsigned int),
                        cudaMemcpyHostToDevice, stream_bkt_HEAD);
        cudaStreamSynchronize(stream_bkt_HEAD);

        // actually move data to buckets
        move_to_buckets<<<blocksPerGrid, threadsPerBlock>>>(
            gpu_from, gpu_to, repr_array_size, gpu_bucket_HEADs,
            gpu_bucket_key_label, gpu_bucket_indexes);

        // FIXME:
        /*entry_repr *debugg = new entry_repr[repr_array_size];
        cudaMemcpy(debugg, gpu_to, repr_array_size * sizeof(unsigned int),
                   cudaMemcpyDeviceToHost);
        for (unsigned int i = 0; i != repr_array_size; ++i) {
          std::cout << debugg[i] << std::endl;
        }*/

        // swap arrays (via pointers {from}/{to} swapping)
        entry_repr *gpu_ptr_swap_tmp = gpu_from;
        gpu_from = gpu_to;
        gpu_to = gpu_ptr_swap_tmp;
    }
    cudaDeviceSynchronize();
    // the correct result is in "gpu_repr_array"

    // Cleanup
    cudaFree(gpu_bucket_indexes);
    cudaFree(gpu_bucket_key_label);
    cudaFree(gpu_bucket_HEADs);

    cudaFreeHost(bucket_indexes);
    cudaFreeHost(bucket_key_label);
    cudaFreeHost(bucket_HEADs);
    cudaFreeHost(frequency);

    cudaStreamDestroy(stream_bkt_idx);
    cudaStreamDestroy(stream_bkt_HEAD);
    cudaStreamDestroy(stream_key_lbl);
}

void encode(entry *gpu_entry_array, entry_repr *gpu_repr_array,
            unsigned int repr_array_size, char (*gpu_result_array)[32]) {
    constexpr const int threadsPerBlock = 1024;
    const int blocksPerGrid =
        (repr_array_size + threadsPerBlock - 1) / threadsPerBlock;

    expand_and_encode<<<blocksPerGrid, threadsPerBlock>>>(
        gpu_entry_array, gpu_repr_array, repr_array_size, gpu_result_array);
}

}  // namespace sort
