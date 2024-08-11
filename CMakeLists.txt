cmake_minimum_required(VERSION 2.8.3)
project(topo_loc)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3")

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

##########
# CPP #
##########
# add_subdirectory(src/cpp)

##########
# EXPORT #
##########
cs_install()
cs_export()
