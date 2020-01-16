add_executable(fmindex main.cpp)


target_sources(fmindex
    PRIVATE
        types.hpp
        types.cpp
        TA.hpp
)

target_sources(fmindex
    PRIVATE
        # cpu, single thread
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,cpu>,radix_sort.hpp,>
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,cpu>,radix_sort.cpp,>
        # cpu, multi thread
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,parallel>,parallel_radix_sort.hpp,>
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,parallel>,parallel_radix_sort.cpp,>
        # gpu, cuda
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,gpu>,gpu_radix_sort.hpp,>
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,gpu>,gpu_radix_sort.cpp,>
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,gpu>,kernels.cuh,>
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,gpu>,kernels.cu,>
)


set_source_files_properties(fmindex
    gpu_radix_sort.hpp
    gpu_radix_sort.cpp
    PROPERTIES LANGUAGE CUDA
)


target_compile_features(fmindex
    PUBLIC
        cxx_std_14
)


set_target_properties(fmindex
    PROPERTIES
        CXX_STANDARD 14
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF

        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 14
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS ON
)


target_compile_options(fmindex
    PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic -O0>
        $<$<COMPILE_LANGUAGE:CUDA>:>
)


if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Linux)
    find_package(Threads REQUIRED)
    target_link_libraries(fmindex Threads::Threads)
endif()