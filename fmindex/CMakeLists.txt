add_executable(fmindex main.cpp)


target_sources(fmindex
    PRIVATE
        types.hpp
        types.cpp
)

target_sources(fmindex
    PRIVATE
        # cpu, single thread
        $<IF:$<BOOL:NOT DEFINED SORT>,radix_sort.hpp,>
        $<IF:$<BOOL:NOT DEFINED SORT>,radix_sort.cpp,>
        # cpu, multi thread
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,parallel>,parallel_radix_sort.hpp,>
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,parallel>,parallel_radix_sort.cpp,>
        # gpu, cuda
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,gpu>,gpu_radix_sort.hpp,>
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,gpu>,radix_sort.cpp,>
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,gpu>,radix_sort.cuh,>
        $<IF:$<STREQUAL:$<LOWER_CASE:${SORT}>,gpu>,radix_sort.cu,>
)


#set_source_files_properties(fmindex
#    main.cu
#    PROPERTIES LANGUAGE CUDA
#)


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
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:>
)
