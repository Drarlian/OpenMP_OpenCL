cmake_minimum_required(VERSION 3.29)
project(A4_Programacao_Paralela C)

set(CMAKE_C_STANDARD 17)

# --- OpenMP (CPU) ---
add_executable(A4_OpenMP openmp.c)
target_compile_options(A4_OpenMP PRIVATE -fopenmp)
target_link_options(A4_OpenMP PRIVATE -fopenmp)

# --- OpenCL (GPU) ---
add_executable(A4_OpenCL opencl.c)

# Headers do OpenCL (Khronos) dentro do projeto
target_include_directories(
        A4_OpenCL PRIVATE
        "${CMAKE_SOURCE_DIR}/external/opencl/include"
)

# Linkar com a cópia local da DLL
target_link_libraries(
        A4_OpenCL PRIVATE
        "${CMAKE_SOURCE_DIR}/external/opencl/bin/OpenCL.dll"
)
