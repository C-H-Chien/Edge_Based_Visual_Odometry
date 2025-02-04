CMAKE_MINIMUM_REQUIRED( VERSION 2.8 )
message(STATUS "CMake version: ${CMAKE_VERSION}")

set(CMAKE_CXX_STANDARD "11")
if(NOT CMAKE_CXX_STANDARD_REQUIRED)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

PROJECT(LEMS_VO_SLAM)

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall" CACHE STRING "" FORCE)

#> Crucial flags for performance
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native" CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -ffast-math" CACHE STRING "" FORCE)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#> Dependencies
FIND_PACKAGE(Eigen3 REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Ceres REQUIRED)
FIND_PACKAGE(OpenMP REQUIRED)
FIND_PACKAGE(CSparse REQUIRED)

#> If you have ceres installed, you already have the following dependencies
FIND_PACKAGE(Glog REQUIRED)
FIND_PACKAGE(CSparse REQUIRED)
FIND_PACKAGE(yaml-cpp REQUIRED)

#FIND_PACKAGE(Pangolin REQUIRED)

message(STATUS "Path is: ’${OpenCV_PATH}'")
message("\n=====OpenCV Library=====")
message(STATUS "[Directory]:    ${OpenCV_DIR}")
message(STATUS "[Version]:      ${OpenCV_VERSION}")
message(STATUS "[Libraries]:    ${OpenCV_LIBS}")
message(STATUS "[Include Path]: ${OpenCV_INCLUDE_DIRS}")
message( "========================\n")

#> YAML Data Parser
include_directories(/usr/local/include/yaml-cpp)
include_directories(${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

set(THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${Eigen_LIBS}
        ${Sophus_LIBRARIES}
        #${Pangolin_LIBRARIES} GL GLU GLEW (FOR LINUX)
        ${Pangolin_LIBRARIES} "-framework OpenGL" "-framework GLUT" ${GLEW_LIBRARIES} #(FOR MACOS)
        #${GTEST_BOTH_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        pthread
        ${CSparse_LIBRARY}
        /usr/local/lib/libyaml-cpp.so
        )

enable_testing()

#> All header files
include_directories(${PROJECT_SOURCE_DIR}/include)

#> Subdirectories
add_subdirectory(src)

add_subdirectory(cmd)
add_subdirectory(test)
