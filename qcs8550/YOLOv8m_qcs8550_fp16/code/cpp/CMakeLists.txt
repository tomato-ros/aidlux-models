cmake_minimum_required(VERSION 3.22.1)
project(aidlite_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MODEL_SRC ${CMAKE_SOURCE_DIR}/data)
set(MODEL_DST ${CMAKE_BINARY_DIR}/data)

add_executable(aidlite_demo src/main.cpp)

target_include_directories(aidlite_demo PRIVATE
        /usr/local/include/aidlux/aidlite
)

target_link_directories(aidlite_demo PRIVATE
        /usr/local/lib
)

target_link_libraries(aidlite_demo PRIVATE
        aidlite
)


file(COPY ${MODEL_SRC}
        DESTINATION ${CMAKE_BINARY_DIR}
)









