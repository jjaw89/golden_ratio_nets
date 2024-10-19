cmake_minimum_required(VERSION 3.16)

if(NOT DEFINED PROJECT_NAME)
  set(STAR_DISCREPANCY_MASTER_PROJECT ON)
else()
  set(STAR_DISCREPANCY_MASTER_PROJECT OFF)
endif()

project(
  STAR_DISCREPANCY
  VERSION 0.1.0
  DESCRIPTION "Algorithms for the Star Discrepancy Problem"
  LANGUAGES C
)

# Project options
option(STAR_DISCREPANCY_DEVELOPMENT_BUILD "Development build" OFF)
option(STAR_DISCREPANCY_ENABLE_WARNINGS "Enable warnings" ${STAR_DISCREPANCY_DEVELOPMENT_BUILD})
option(STAR_DISCREPANCY_ENABLE_WERROR "Enable -Werror or /WX" OFF)

# Warning flags
set(STAR_DISCREPANCY_C_WARN_FLAGS "")
if (STAR_DISCREPANCY_ENABLE_WARNINGS)
  if (CMAKE_C_COMPILER_ID MATCHES "MSVC")
    list(APPEND STAR_DISCREPANCY_C_WARN_FLAGS /W4)
  elseif(CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    list(APPEND STAR_DISCREPANCY_C_WARN_FLAGS
      -Wall
      -Wextra
      -Wpedantic
      -pedantic-errors
      -Wundef
      -Wredundant-decls
      -Wcast-qual
      -Wmissing-include-dirs
      -Wdisabled-optimization
      -Wconversion
      -Wdouble-promotion
      -Wshadow
      -Wnull-dereference
    )
    if(CMAKE_C_COMPILER_ID MATCHES "GNU")
      list(APPEND STAR_DISCREPANCY_C_WARN_FLAGS
        -Wduplicated-cond
      )
    endif()
  else()
    message(WARNING "Unknown compiler, not setting any warning flags.")
    set(STAR_DISCREPANCY_C_WARN_FLAGS "")
  endif()
endif()

if(STAR_DISCREPANCY_ENABLE_WERROR)
  list(APPEND STAR_DISCREPANCY_C_WARN_FLAGS
    $<$<C_COMPILER_ID:GNU,Clang>:-Werror>
    $<$<C_COMPILER_ID:MSVC>:/WX>
  )
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

# Extra stuff if this is the master project
if(STAR_DISCREPANCY_MASTER_PROJECT)
  # Turn off C extensions
  set(CMAKE_C_EXTENSIONS OFF)
  # Include CTest
  include(CTest)
  # And if this is a development build
  if(STAR_DISCREPANCY_DEVELOPMENT_BUILD)
    # Create compile commands json for clang tools
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    set(CMAKE_C_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_C_IMPLICIT_INCLUDE_DIRECTORIES})
  endif()
endif()

include(GNUInstallDirs)

find_package(OpenMP REQUIRED)

file(GLOB_RECURSE sources CONFIGURE_DEPENDS "src/*.c")
add_library(sdiscr ${sources})
target_link_libraries(sdiscr m gsl blas OpenMP::OpenMP_C)
target_compile_features(sdiscr PRIVATE c_std_11)
target_compile_options(sdiscr PRIVATE ${STAR_DISCREPANCY_C_WARN_FLAGS})
target_include_directories(sdiscr PUBLIC 
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

add_executable(sdiscr_pointset apps/sdiscr_pointset.c)
target_link_libraries(sdiscr_pointset sdiscr)
target_compile_features(sdiscr_pointset PRIVATE c_std_11)
target_compile_options(sdiscr_pointset PRIVATE ${STAR_DISCREPANCY_C_WARN_FLAGS})

add_executable(sdiscr_simple apps/sdiscr_simple.c)
target_link_libraries(sdiscr_simple sdiscr)
target_compile_features(sdiscr_simple PRIVATE c_std_11)
target_compile_options(sdiscr_simple PRIVATE ${STAR_DISCREPANCY_C_WARN_FLAGS})

add_executable(sdiscr_bz apps/sdiscr_bz.c)
target_link_libraries(sdiscr_bz sdiscr)
target_compile_features(sdiscr_bz PRIVATE c_std_11)
target_compile_options(sdiscr_bz PRIVATE ${STAR_DISCREPANCY_C_WARN_FLAGS})

add_executable(sdiscr_dem apps/sdiscr_dem.c)
target_link_libraries(sdiscr_dem sdiscr)
target_compile_features(sdiscr_dem PRIVATE c_std_11)
target_compile_options(sdiscr_dem PRIVATE ${STAR_DISCREPANCY_C_WARN_FLAGS})

add_executable(sdiscr_dem_parallel apps/sdiscr_dem_parallel.c)
target_link_libraries(sdiscr_dem_parallel sdiscr)
target_compile_features(sdiscr_dem_parallel PRIVATE c_std_11)
target_compile_options(sdiscr_dem_parallel PRIVATE ${STAR_DISCREPANCY_C_WARN_FLAGS})

if(STAR_DISCREPANCY_DEVELOPMENT_BUILD)
  set(STAR_DISCREPANCY_ORIGINAL_C_WARN_FLAGS "")
  list(APPEND STAR_DISCREPANCY_ORIGINAL_C_WARN_FLAGS
    -Wno-incompatible-pointer-types
    -Wno-unused-result
  )

  add_executable(original_simple_discr original/simple_main.c original/simple_discr.c)
  target_compile_features(original_simple_discr PRIVATE c_std_11)
  target_compile_options(original_simple_discr PRIVATE ${STAR_DISCREPANCY_ORIGINAL_C_WARN_FLAGS})

  add_executable(original_bz_discr original/bz_main.c original/bz_discr.c)
  target_compile_features(original_bz_discr PRIVATE c_std_11)
  target_compile_options(original_bz_discr PRIVATE ${STAR_DISCREPANCY_ORIGINAL_C_WARN_FLAGS})

  add_executable(original_dem_discr original/dem_main.c original/dem_discr.c original/bz_discr.c)
  target_link_libraries(original_dem_discr m)
  target_compile_features(original_dem_discr PRIVATE c_std_11)
  target_compile_options(original_dem_discr PRIVATE ${STAR_DISCREPANCY_ORIGINAL_C_WARN_FLAGS})

  # Generate golden tests data
  add_custom_command(
    TARGET sdiscr_pointset
    POST_BUILD
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/gen_data.py
      "${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/data"
      "$<TARGET_FILE:sdiscr_pointset>"
  )
  add_custom_command(
    TARGET sdiscr_simple
    POST_BUILD
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/gen_outputs.py
      "${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/data"
      "$<TARGET_FILE:sdiscr_simple>"
  )
  # Golden tests
  add_test(
    NAME golden-test-sdiscr_simple
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/test_executable.py
      "${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/data"
      "$<TARGET_FILE:sdiscr_simple>"
  )
  add_test(
    NAME golden-test-sdiscr_bz
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/test_executable.py
      "${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/data"
      "$<TARGET_FILE:sdiscr_bz>"
  )
  add_test(
    NAME golden-test-sdiscr_dem
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/test_executable.py
      "${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/data"
      "$<TARGET_FILE:sdiscr_dem>"
  )
  add_test(
    NAME golden-test-sdiscr_dem_parallel
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/test_executable.py
      "${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/data"
      "$<TARGET_FILE:sdiscr_dem_parallel>"
  )
  add_test(
    NAME golden-test-original_simple_discr
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/test_executable.py
      "${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/data"
      "$<TARGET_FILE:original_simple_discr>"
  )
  add_test(
    NAME golden-test-original_bz_discr
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/test_executable.py
      "${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/data"
      "$<TARGET_FILE:original_bz_discr>"
  )
  add_test(
    NAME golden-test-original_dem_discr
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/test_executable.py
      "${CMAKE_CURRENT_SOURCE_DIR}/test/golden_tests/data"
      "$<TARGET_FILE:original_dem_discr>"
  )
endif()
