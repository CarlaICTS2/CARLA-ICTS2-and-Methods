cmake_minimum_required(VERSION 3.8)

project(is_despot)

set(BINARY_INSTALL_PATH "bin" CACHE PATH "Binary install path")
set(LIBRARY_INSTALL_PATH "lib" CACHE PATH "Library install path")
set(INCLUDE_INSTALL_PATH "include" CACHE PATH "Include install path")
set(CONFIG_INSTALL_PATH "${LIBRARY_INSTALL_PATH}/${PROJECT_NAME}/cmake")

set(DESPOT_BUILD_EXAMPLES ON CACHE BOOL "Build C++ model examples")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2 -mfpmath=sse -std=c++17")
set(CMAKE_MODULE_PATH ${CMAKE_PREFIX_PATH} "${PROJECT_SOURCE_DIR}/cmake")

include_directories(include)

add_library("${PROJECT_NAME}" SHARED
  src/interface/pomdp.cpp
  src/interface/world.cpp
  src/interface/belief.cpp
  src/interface/lower_bound.cpp
  src/interface/upper_bound.cpp
  src/interface/default_policy.cpp

  src/core/builtin_policy.cpp
  src/core/globals.cpp
  src/core/node.cpp
  src/core/particle_belief.cpp
  src/core/solver.cpp

  src/random_streams.cpp
  src/solver/despot.cpp

  src/util/coord.cpp
  src/util/logging.cpp
  src/util/random.cpp
  src/util/seeds.cpp
  src/util/util.cpp
)

# Build example files
if(DESPOT_BUILD_EXAMPLES)
  add_subdirectory(carla_car)
endif()

install(TARGETS "${PROJECT_NAME}"
  EXPORT "DespotTargets"
  ARCHIVE DESTINATION "${LIBRARY_INSTALL_PATH}"
  LIBRARY DESTINATION "${LIBRARY_INSTALL_PATH}"
  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
)

install(DIRECTORY "include/${PROJECT_NAME}/"
  DESTINATION "${INCLUDE_INSTALL_PATH}/${PROJECT_NAME}"
)

# Install a DespotConfig.cmake file so CMake can find_package(Despot).
include(CMakePackageConfigHelpers)
configure_package_config_file("cmake/DespotConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/DespotConfig.cmake"
  INSTALL_DESTINATION "${CONFIG_INSTALL_PATH}"
  PATH_VARS INCLUDE_INSTALL_PATH
)

install(EXPORT "DespotTargets"
  FILE "DespotTargets.cmake"
  DESTINATION "${CONFIG_INSTALL_PATH}"
)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/DespotConfig.cmake"
  DESTINATION "${CONFIG_INSTALL_PATH}"
)

