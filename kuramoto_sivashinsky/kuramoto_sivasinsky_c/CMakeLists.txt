cmake_minimum_required(VERSION 3.3.0)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(WARNING_CXX_FLAGS "-Wall -Werror -pedantic -pedantic-errors")
set(WARNING_CXX_FLAGS "${WARNING_CXX_FLAGS} -Wextra -Wunreachable-code")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} ${WARNING_CXX_FLAGS} -march=native -std=gnu++17")

project(adjoint_march)

set(SOURCE_FILES
    main.cpp
    adjoint_march.cpp
    ks_equations.cpp)

add_executable(adjoint_march ${SOURCE_FILES})
