cmake_minimum_required(VERSION 3.25)

find_package(cmake-bare REQUIRED PATHS node_modules/cmake-bare)
find_package(cmake-vcpkg REQUIRED PATHS node_modules/cmake-vcpkg)

project(classifier-addon C CXX)

find_path(QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS "inference-addon-cpp/JsInterface.hpp")

add_bare_module(classifier-addon)

target_sources(
  ${classifier-addon}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/addon/binding.cpp
)

target_include_directories(
  ${classifier-addon}
  SYSTEM PRIVATE
    ${QVAC_LIB_INFERENCE_ADDON_CPP_INCLUDE_DIRS}
)

target_include_directories(
  ${classifier-addon}
  PRIVATE
    ${PROJECT_SOURCE_DIR}/addon
)
