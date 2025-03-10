cmake_minimum_required(VERSION 3.8)

add_executable("${PROJECT_NAME}_carla_car"
  src/collision.cpp
  src/connector.h
  src/coord.h
  src/math_utils.cpp
  src/math_utils.h
  src/param.cpp
  src/param.h
  src/path.cpp
  src/path.h
  src/ped_pomdp.cpp
  src/ped_pomdp.h
  src/simulator.cpp
  src/state.h
  src/world_model.cpp
  src/world_model.h
  src/optionparser.h
)

target_link_libraries("${PROJECT_NAME}_carla_car"
  "${PROJECT_NAME}"
)
install(TARGETS "${PROJECT_NAME}_carla_car"
  RUNTIME DESTINATION "${BINARY_INSTALL_PATH}"
)