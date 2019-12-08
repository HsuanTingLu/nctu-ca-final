add_executable(fmindex)


target_sources(fmindex
    PRIVATE
        types.hpp
        radix_sort.hpp

        main.cpp
        types.cpp
        radix_sort.cpp
)


#set_source_files_properties(fmindex
#    main.cu
#    PROPERTIES LANGUAGE CUDA
#)


target_compile_features(fmindex
    PUBLIC
        c_std_11
        cxx_std_14
)


set_target_properties(fmindex
    PROPERTIES
        C_STANDARD 11
        C_STANDARD_REQUIRED ON
        C_EXTENSIONS OFF

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
        #$<$<COMPILE_LANGUAGE:CUDA>:>
)
