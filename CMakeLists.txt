cmake_minimum_required(VERSION 3.15)
project(Colisiones)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)

find_package(glfw3 REQUIRED)

find_path(GLM_INCLUDE_DIR glm/glm.hpp
    HINTS 
    /opt/homebrew/include 
)

set(IMGUI_DIR "${CMAKE_SOURCE_DIR}/third_party/imgui")

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)

add_executable(colisiones
    main.cpp 
    ${IMGUI_SOURCES}
)

target_include_directories(colisiones PRIVATE
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${GLM_INCLUDE_DIR}
    /opt/homebrew/include 
)

target_link_libraries(colisiones PRIVATE
    glfw
    OpenGL::GL
)