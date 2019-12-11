cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

if(NOT ${UNIX})
    message(FATAL_ERROR " not intended for non-UNIX systems")
endif()

project(howhow LANGUAGES CXX)


if(NOT DEFINED SORT)
    set(SORT cpu)
    message(WARNING " using single threaded radix sort")
elseif(${SORT} STREQUAL parallel)
    message(WARNING " using multi-thread radix sort")
elseif(${SORT} STREQUAL gpu)
    enable_language(CUDA)
    message(WARNING " using GPU radix sort")
else()
    message(FATAL_ERROR " invalid sorting config")
endif()


set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
add_subdirectory(fmindex)
