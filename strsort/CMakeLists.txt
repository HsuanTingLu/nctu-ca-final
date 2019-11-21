
add_executable(strsort)


target_sources(strsort
    PUBLIC
    PRIVATE
        main.cpp
)


set_source_files_properties(strsort
    main.cpp 
    PROPERTIES LANGUAGE CXX
)


set_source_files_properties(strsort
    xxx.cu
    PROPERTIES LANGUAGE CUDA
)


target_compile_features(strsort
    PUBLIC
        c_std_11
        cxx_std_14
)


set_target_properties(strsort
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


target_compile_options(strsort
    PUBLIC
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -pedantic>
        $<$<COMPILE_LANGUAGE:CUDA>:>)
)
