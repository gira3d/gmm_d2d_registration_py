cmake_minimum_required(VERSION 3.0.2)
project(gmm_d2d_registration_py)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

find_package(pybind11 REQUIRED)
find_package(gmm_d2d_registration REQUIRED)
find_package(gmm REQUIRED)
find_package(dlib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost COMPONENTS filesystem REQUIRED)

set(CMAKE_CXX_STANDARD 17)

include_directories(
  ${gmm_d2d_registration_INCLUDE_DIR}
  ${gmm_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIRS}
  ${dlib_INCLUDE_DIR}
  ${pybind11_INCLUDE_DIR}
  ${Boost_INCLUDE_DIR}
)

# pybind modules
pybind11_add_module(${PROJECT_NAME}
  src/${PROJECT_NAME}/${PROJECT_NAME}.cpp
)

target_link_libraries(${PROJECT_NAME}
  PRIVATE
  ${PYTHON_LIBRARIES}
  ${BOOST_LIBRARIES}
  ${Boost_FILESYSTEM_LIBRARY}
  ${Boost_SYSTEM_LIBRARY}
  dlib::dlib
)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY
  ${CMAKE_INSTALL_PREFIX}
)

install(TARGETS DESTINATION ${CMAKE_INSTALL_PREFIX})
