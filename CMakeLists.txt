cmake_minimum_required(VERSION 3.10)
project(ARForecasting)

set(CMAKE_CXX_STANDARD 11)

add_executable(ARForecasting 
    src/main.cpp 
    src/ARModel.cpp 
    src/SyntheticDataGenerator.cpp
)
