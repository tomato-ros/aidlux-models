cmake_minimum_required(VERSION 3.22.1)
project(aidlite_demo)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(MODEL_SRC ${CMAKE_SOURCE_DIR}/data)
set(MODEL_DST ${CMAKE_BINARY_DIR}/data)

add_executable(aidlite_demo src/main.cpp)

target_include_directories(aidlite_demo PRIVATE
        /usr/local/include/aidlux/aidlite
        /usr/include/opencv4
)

target_link_directories(aidlite_demo PRIVATE
        /usr/local/lib
)

target_link_libraries(aidlite_demo PRIVATE
        aidlite
        opencv_core
        opencv_imgproc
        opencv_imgcodecs
        pthread
)


file(COPY ${MODEL_SRC}
        DESTINATION ${CMAKE_BINARY_DIR}
)









