cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

if(NOT ${UNIX})
    message(FATAL_ERROR " not intended for non-UNIX systems")
endif()

project(howhow LANGUAGES CXX CUDA C)

add_subdirectory(fmindex)
