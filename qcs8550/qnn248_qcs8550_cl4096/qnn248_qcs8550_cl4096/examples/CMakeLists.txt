cmake_minimum_required(VERSION 3.10)
project(Demo VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    /usr/include/opencv4
)

set(link_libs   aidgen
                opencv_core
                opencv_imgproc
                opencv_imgcodecs
                pthread
)

# add_executable(test_t2t test_aidgen_t2t.cpp)
# target_link_libraries(test_t2t ${link_libs})
# set_target_properties(test_t2t PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
# )

# add_executable(test_t2e test_aidgen_t2e.cpp)
# target_link_libraries(test_t2e ${link_libs})
# set_target_properties(test_t2e PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
# )

# add_executable(test_qwen25vl qwen25vl_test.cpp qwen25vl_image_process.cpp)
# target_link_libraries(test_qwen25vl ${link_libs} aidlite)
# set_target_properties(test_qwen25vl PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
# )

add_executable(test_qwen3vl qwen3vl_test.cpp qwen3vl_image_process.cpp)
target_link_libraries(test_qwen3vl ${link_libs} aidlite)
set_target_properties(test_qwen3vl PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# add_executable(qwen3vl_camera qwen3vl_camera.cpp qwen3vl_image_process.cpp)
# target_link_libraries(qwen3vl_camera ${link_libs} aidlite opencv_videoio)
# set_target_properties(qwen3vl_camera PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
# )

# add_executable(test_locateanything locateanything_test.cpp locateanything_image_process.cpp)
# target_link_libraries(test_locateanything ${link_libs} aidlite)
# set_target_properties(test_locateanything PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
# )