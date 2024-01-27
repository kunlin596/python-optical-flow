cmake_minimum_required(VERSION 3.15)

project(OpticalFlow)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS filesystem system program_options)
find_package(Eigen3 REQUIRED)

add_library(flow SHARED src/flow.cc)
target_link_libraries(flow ${OpenCV_LIBS} ${Boost_LIBRARIES} Eigen3::Eigen)

add_executable(run_optical_flow src/cli.cc)
target_link_libraries(run_optical_flow flow)
