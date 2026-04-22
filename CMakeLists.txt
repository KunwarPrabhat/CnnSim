cmake_minimum_required(VERSION 3.10)
project(MetalNet)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options(/arch:AVX2)
else()
    add_compile_options(-mavx2 -mfma)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include/MetalNet)
include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(MetalNet main.cpp)
add_executable(MetalNetTests tests.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MetalNet PRIVATE OpenMP::OpenMP_CXX)
    target_link_libraries(MetalNetTests PRIVATE OpenMP::OpenMP_CXX)
endif()
