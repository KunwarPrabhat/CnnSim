# =====================================================================
#  MetalNet Profiler targets
# =====================================================================

# --- Self-test: architecture-independent, validates profiler core ----
add_executable(metalnet_profiler_selftest selftest.cpp)
target_compile_definitions(metalnet_profiler_selftest PRIVATE METALNET_PROFILE)

# --- Headless demo: trains real engine, exports Perfetto trace -------
#     Requires a SIMD backend (AVX2 or NEON).
if(MN_SIMD)
    add_executable(metalnet_profiler_demo profiler_demo.cpp)
    target_compile_definitions(metalnet_profiler_demo PRIVATE METALNET_PROFILE)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(metalnet_profiler_demo PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# --- Live GUI dashboard (Dear ImGui + ImPlot + GLFW) -----------------
option(METALNET_BUILD_GUI "Build the ImGui/ImPlot profiler GUI" OFF)

if(METALNET_BUILD_GUI AND MN_SIMD)
    include(FetchContent)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)

    # OpenGL
    set(OpenGL_GL_PREFERENCE GLVND)
    find_package(OpenGL REQUIRED)

    # GLFW: prefer system install, else fetch
    find_package(glfw3 3.3 QUIET)
    if(NOT glfw3_FOUND)
        FetchContent_Declare(glfw
            GIT_REPOSITORY https://github.com/glfw/glfw.git
            GIT_TAG        3.4)
        set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
        set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(glfw)
    endif()

    # Dear ImGui (no CMake project -> build sources ourselves)
    FetchContent_Declare(imgui
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG        v1.91.5)
    FetchContent_MakeAvailable(imgui)

    # ImPlot (depends on ImGui)
    FetchContent_Declare(implot
        GIT_REPOSITORY https://github.com/epezent/implot.git
        GIT_TAG        v0.16)
    FetchContent_MakeAvailable(implot)

    add_library(imgui_implot STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
        ${implot_SOURCE_DIR}/implot.cpp
        ${implot_SOURCE_DIR}/implot_items.cpp)
    target_include_directories(imgui_implot PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
        ${implot_SOURCE_DIR})
    target_link_libraries(imgui_implot PUBLIC glfw OpenGL::GL)

    add_executable(metalnet_profiler_gui profiler_gui.cpp)
    target_compile_definitions(metalnet_profiler_gui PRIVATE METALNET_PROFILE)
    target_link_libraries(metalnet_profiler_gui PRIVATE imgui_implot)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(metalnet_profiler_gui PRIVATE OpenMP::OpenMP_CXX)
    endif()
elseif(METALNET_BUILD_GUI AND NOT MN_SIMD)
    message(WARNING "METALNET_BUILD_GUI requested but no SIMD backend for this arch. GUI skipped.")
endif()
