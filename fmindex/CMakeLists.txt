
add_executable(fmindex)


target_sources(fmindex
    PUBLIC
    PRIVATE
        main.cu

        types.hpp

        types.cpp
)


set_source_files_properties(fmindex
    types.cpp 
    PROPERTIES LANGUAGE CXX
)


set_source_files_properties(fmindex
    main.cu
    types.hpp
    PROPERTIES LANGUAGE CUDA
)


#set_source_files_properties(fmindex
#    xxx.c
#    PROPERTIES LANGUAGE C
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
        $<$<COMPILE_LANGUAGE:CUDA>:>
)
