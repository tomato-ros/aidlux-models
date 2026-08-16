cmake_minimum_required(VERSION 3.22.1)

project(aidlite_demo)

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(aidlite_demo src/main.cpp)

target_include_directories(aidlite_demo PRIVATE /usr/local/include/aidlux/aidlite)

target_link_directories(aidlite_demo PRIVATE /usr/local/lib)

target_link_libraries(aidlite_demo PRIVATE aidlite)
