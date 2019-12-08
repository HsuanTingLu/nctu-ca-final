cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

if(NOT ${UNIX})
    message(FATAL_ERROR " not intended for non-UNIX systems")
endif()

project(howhow LANGUAGES CXX C)
#enable_language(CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
add_subdirectory(fmindex)
