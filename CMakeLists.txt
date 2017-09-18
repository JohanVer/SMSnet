project(caffe_test)
cmake_minimum_required(VERSION 2.8)
add_definitions(-DUSE_OPENCV)
add_definitions(${Caffe_DEFINITIONS})

include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
        message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED )
find_package(Boost COMPONENTS system filesystem chrono timer REQUIRED)
find_package(HDF5 REQUIRED)
find_package(Caffe REQUIRED)
find_package(Protobuf REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} ${Caffe_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${CAFFE_BUILD_INCLUDE} ${PROTOBUF_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})

add_library(caffe_test
   src/classify.cpp
   src/flow_color_coder.cpp
   src/hdf5_reader.cpp
   include/classify.h
   include/flow_color_coder.h
   include/hdf5_reader.h
 )

target_link_libraries ( caffe_test ${SYSTEM_LIBS} ${OpenCV_LIBS} ${Boost_LIBRARIES} ${Caffe_LIBRARIES} ${PROTOBUF_LIBRARIES} ${HDF5_LIBRARIES} rt dl z m X11)

add_executable(caffe_test_ex src/main.cpp)
target_link_libraries (caffe_test_ex ${SYSTEM_LIBS} caffe_test ${GLOG_LIBRARIES} ${PROTOBUF_LIBRARIES} ${HDF5_LIBRARIES} rt dl m X11)

