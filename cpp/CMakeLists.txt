cmake_minimum_required(VERSION 3.16)

project(pert_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(pert_mutator
    src/mutator.cpp
)
target_include_directories(pert_mutator
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(pert_te
    src/te.cpp
)
target_include_directories(pert_te
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(pert_mutator_cli
    src/mutator_cli.cpp
)
target_link_libraries(pert_mutator_cli PRIVATE pert_mutator)

enable_testing()

add_executable(test_mutator
    tests/test_mutator.cpp
)
target_link_libraries(test_mutator PRIVATE pert_mutator)
add_test(NAME mutator_parity COMMAND test_mutator)

add_executable(test_te
    tests/test_te.cpp
)
target_link_libraries(test_te PRIVATE pert_te)
add_test(NAME te_directionality COMMAND test_te)
