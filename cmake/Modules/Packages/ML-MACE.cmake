cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

message("Hello from ML-MACE.cmake.")

find_package(Torch REQUIRED)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
#target_include_directories(lammps PRIVATE "${TORCH_INCLUDE_DIRS}")
target_link_libraries(lammps PRIVATE "${TORCH_LIBRARIES}")
set_property(TARGET lammps PROPERTY CXX_STANDARD 14)
