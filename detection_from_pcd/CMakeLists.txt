cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(training)

find_package(PCL 1.2 REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (training src/training.cpp)
target_link_libraries (training ${PCL_LIBRARIES})
