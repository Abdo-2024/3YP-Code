cmake_minimum_required(VERSION 3.14.0)

project(SWN)

### Configure output locations
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)  

# Print the build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release" FORCE)
endif()
message(STATUS "cmake build type: ${CMAKE_BUILD_TYPE}")

### Configure the compiler
# This is a basic, decent setup that should do something sane on most compilers

if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")

  # using Clang (linux or apple) or GCC
  message("Using clang/gcc compiler flags")
  SET(CMAKE_CXX_STANDARD 17)
  SET(BASE_CXX_FLAGS "-std=c++17 -Wall -Wextra") 
  SET(DISABLED_WARNINGS "-Wno-unused-label -Wno-unused-local-typedef -Wno-register -Wno-sign-compare -Wno-unused-parameter -Wno-unused-but-set-parameter -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function -Wno-deprecated-declarations -Wno-missing-braces -Wno-unused-private-field")
  SET(TRACE_INCLUDES " -H -Wno-error=unused-command-line-argument")

  if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    message("Setting clang-specific options")
    SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -ferror-limit=3 -Wfatal-errors -fcolor-diagnostics")
    SET(CMAKE_CXX_FLAGS_DEBUG          "-g3 -fsanitize=address -fno-limit-debug-info")
  elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    message("Setting gcc-specific options")
    SET(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} -fmax-errors=5")
    SET(CMAKE_CXX_FLAGS_DEBUG          "-g3")
    SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} -Wno-maybe-uninitialized -Wno-format-zero-length -Wno-unused-but-set-parameter -Wno-unused-but-set-variable")
  endif()
  
  SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS} -std=c++17")

  include(CheckCXXCompilerFlag)
  unset(COMPILER_SUPPORTS_MARCH_NATIVE CACHE)
  CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
  if(COMPILER_SUPPORTS_MARCH_NATIVE)
    SET(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG -std=c++17")
  else()
    SET(CMAKE_CXX_FLAGS_RELEASE        "-O3 -mcpu=apple-m1 -DNDEBUG -std=c++17") # Apple M1
  endif()

elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
  # using Visual Studio C++
  message("Using Visual Studio compiler flags")
  set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /W4")
  set(BASE_CXX_FLAGS "${BASE_CXX_FLAGS} /MP") # parallel build
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4267\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4244\"")  # ignore conversion to smaller type (fires more aggressively than the gcc version, which is annoying)
  SET(DISABLED_WARNINGS "${DISABLED_WARNINGS} /wd\"4305\"")  # ignore truncation on initialization
  SET(CMAKE_CXX_FLAGS "${BASE_CXX_FLAGS} ${DISABLED_WARNINGS}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")

  add_definitions(/D "_CRT_SECURE_NO_WARNINGS")
  add_definitions(-DNOMINMAX)
  add_definitions(-D_USE_MATH_DEFINES)
else()
  # unrecognized
  message( FATAL_ERROR "Unrecognized compiler [${CMAKE_CXX_COMPILER_ID}]" )
endif()

# == Deps
add_subdirectory(deps/geometry-central)
add_subdirectory(deps/polyscope)

find_package(GUROBI)
if (GUROBI_FOUND)
  message(STATUS "Gurobi found at ${GUROBI_LIBRARY}")
else ()
  message(STATUS "Gurobi not found.")
endif ()
include_directories(${GUROBI_INCLUDE_DIRS})

find_package(ortools CONFIG REQUIRED)
set(BUILD_DEPS OFF)

# == Build our project stuff

message("Setting src...")
set(SRCS 
  src/utils.cpp
  src/surface_winding_numbers.cpp
  src/main.cpp
)

add_executable(main "${SRCS}")
target_include_directories(main PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include/")

target_compile_options(main PUBLIC -Wno-unused-parameter -Wno-deprecated-register -std=c++17)
target_compile_features(main PUBLIC cxx_std_17)

# add the args.hxx project which we use for command line args
target_include_directories(main PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/polyscope/deps/args")
target_link_libraries(main geometry-central polyscope ortools::ortools)
message(STATUS "${GUROBI_HOME}")
message(STATUS "${GUROBI_CXX_LIBRARY}")
message(STATUS "${GUROBI_CXX_DEBUG_LIBRARY}")
target_link_libraries(main optimized ${GUROBI_CXX_LIBRARY} debug ${GUROBI_CXX_DEBUG_LIBRARY})
target_link_libraries(main ${GUROBI_LIBRARY})