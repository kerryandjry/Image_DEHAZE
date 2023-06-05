cmake_minimum_required(VERSION 3.12)
project(DEHAZE LANGUAGES CXX)


find_package(OpenCV)

add_executable(Blurimg src/BlurImg.cpp src/mytime.cpp)

target_link_libraries(Blurimg ${OpenCV_LIBS})
