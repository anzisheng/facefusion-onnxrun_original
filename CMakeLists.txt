cmake_minimum_required(VERSION 3.20)
project(face_cpp)
set(cmake_build_type debug)
set(CMAKE_CXX_STANDARD 17)
find_package(OpenCV REQUIRED)

set(ONNXRUNTIME_ROOT "/opt/onnxruntime")

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
PATHS ${ONNXRUNTIME_ROOT}/include
NO_DEFAULT_PATH)

find_package(FMT REQUIRED)
find_package(jsoncpp REQUIRED)

find_library(ONNXRUNTIME_LIBRARY
             NAMES onnxruntime
             PATHS ${ONNXRUNTIME_ROOT}/lib
             NO_DEFAULT_PATH)

# 检查是否找到 ONNX Runtime
if (NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime not found. Please set ONNXRUNTIME_ROOT to the correct path.")
endif()

add_library(cpp_api
        STATIC                     #SHARED
        #src/engine.cpp
        cpp/yolov8face.cpp
        cpp/utils.cpp
        #cpp/utile.cpp
        cpp/face68landmarks.cpp
        cpp/facerecognizer.cpp
        cpp/faceswap.cpp
        cpp/faceenhancer.cpp
        #src/json_info.cpp
        )


include_directories(/usr/include/jsoncpp)
target_include_directories(cpp_api PUBLIC ${OpenCV_INCLUDE_DIRS} ${ONNXRUNTIME_INCLUDE_DIR} cpp/ .)

target_link_libraries(cpp_api PUBLIC ${OpenCV_LIBS}  ${CMAKE_THREAD_LIBS_INIT} ${ONNXRUNTIME_LIBRARY}  fmt::fmt)
# 包含 ONNX Runtime 头文件
#target_include_directories(face_cpp PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})

add_executable(face_cpp cpp/main.cpp)
# 链接 ONNX Runtime 库
target_link_libraries(face_cpp PUBLIC cpp_api  fmt::fmt -lssl -lcrypto jsoncpp_lib )
