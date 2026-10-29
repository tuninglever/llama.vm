cmake_minimum_required(VERSION 3.16)
project(llama-vm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

# path to the llama.cpp tree providing headers and the prebuilt ggml libraries.
# Point this at your llama.cpp checkout, e.g. -DLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp
set(LlamaVM_LLAMA_CPP_DIR "" CACHE PATH "path to the llama.cpp source tree")
if (NOT LlamaVM_LLAMA_CPP_DIR)
    message(FATAL_ERROR "set -DLlamaVM_LLAMA_CPP_DIR=/path/to/llama.cpp")
endif ()

add_library(llama_vm SHARED
    src/pager.cpp
    src/vram.cpp
    src/gguf.cpp
    src/ggml_buffer.cpp
    src/ggml_buffer_gpu.cpp
    src/forthllm.cpp
    src/jl_projector.cpp
    src/bwt.cpp
    src/fm_index.cpp
    src/squeezer.cpp)
target_include_directories(llama_vm
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LlamaVM_LLAMA_CPP_DIR}/ggml/include
        ${LlamaVM_LLAMA_CPP_DIR}/ggml/src)

find_library(LLAMA_VM_GGML_BASE
    NAMES ggml-base
    PATHS ${LlamaVM_LLAMA_CPP_DIR}/build/bin
    NO_DEFAULT_PATH)
find_library(LLAMA_VM_GGML
    NAMES ggml
    PATHS ${LlamaVM_LLAMA_CPP_DIR}/build/bin
    NO_DEFAULT_PATH)
if (NOT LLAMA_VM_GGML_BASE)
    message(FATAL_ERROR "libggml-base.so not found under ${LlamaVM_LLAMA_CPP_DIR}/build/bin")
endif ()
if (NOT LLAMA_VM_GGML)
    message(FATAL_ERROR "libggml.so not found under ${LlamaVM_LLAMA_CPP_DIR}/build/bin")
endif ()
message(STATUS "llama.vm: using ${LLAMA_VM_GGML_BASE} and ${LLAMA_VM_GGML}")

find_package(Threads REQUIRED)
target_link_libraries(llama_vm
    PUBLIC
        ${LLAMA_VM_GGML_BASE}
        ${LLAMA_VM_GGML}
    PRIVATE
        Threads::Threads)
if (CMAKE_SYSTEM_NAME STREQUAL "Darwin")
    target_link_options(llama_vm PRIVATE "-Wl,-undefined,dynamic_lookup")
endif ()

# ------------------------------ CLI tool --------------------------------

add_executable(llama-vm tools/llama-vm.cpp)
target_link_libraries(llama-vm PRIVATE llama_vm)
target_include_directories(llama-vm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${LlamaVM_LLAMA_CPP_DIR}/ggml/include)

# ------------------------------ tests -----------------------------------

enable_testing()

set(LLAMA_VM_TESTS
    test_arena
    test_pager
    test_concurrent
    test_ggml_buffer
    test_gguf
    test_vram
    test_captured
    test_gpu_buffer
    test_forthllm
    test_squeezer)

# the build tree must shadow any installed libllama_vm.so (an old copy in
# /usr/local/lib would otherwise win via LD_LIBRARY_PATH, which outranks the
# binaries' RUNPATH); the rest of the library path is kept as-is
set(LLAMA_VM_TEST_ENV "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}:$ENV{LD_LIBRARY_PATH}")

foreach (t IN LISTS LLAMA_VM_TESTS)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE llama_vm)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${LlamaVM_LLAMA_CPP_DIR}/ggml/include)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES ENVIRONMENT "${LLAMA_VM_TEST_ENV}")
endforeach ()

# integration test: load a real model through llama.cpp with the LLAMA_VM buffer
# (needs the core libllama; auto-skips at run time if no model is available)
find_library(LLAMA_VM_LLAMA
    NAMES llama
    PATHS ${LlamaVM_LLAMA_CPP_DIR}/build/bin
    NO_DEFAULT_PATH)
if (LLAMA_VM_LLAMA)
    add_executable(test_llama_integration tests/test_llama_integration.cpp)
    target_link_libraries(test_llama_integration PRIVATE llama_vm ${LLAMA_VM_LLAMA})
    target_include_directories(test_llama_integration PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${LlamaVM_LLAMA_CPP_DIR}/include
        ${LlamaVM_LLAMA_CPP_DIR}/ggml/include)
    add_test(NAME test_llama_integration COMMAND test_llama_integration)
    set_tests_properties(test_llama_integration PROPERTIES ENVIRONMENT "${LLAMA_VM_TEST_ENV}")
else ()
    message(STATUS "llama.vm: libllama not found; skipping test_llama_integration")
endif ()

# ------------------- standalone ggml backend (dlopen) ---------------------
# thin shim so llama.cpp can load the buffer backend via the standard ggml
# backend path (ggml_backend_load / GGML_BACKEND_PATH); the backend itself
# (and its single state) lives in libllama_vm.so
add_library(ggml-llamavm-cpu SHARED src/ggml_backend_shim.cpp)
target_link_libraries(ggml-llamavm-cpu PRIVATE llama_vm)
target_include_directories(ggml-llamavm-cpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LlamaVM_LLAMA_CPP_DIR}/ggml/include
    ${LlamaVM_LLAMA_CPP_DIR}/ggml/src)
set_target_properties(ggml-llamavm-cpu PROPERTIES PREFIX "lib" OUTPUT_NAME "ggml-llamavm-cpu")

# dlopen-able shim for the GPU weight buffer + wrapper backend (LLAMA_VM_GPU);
# llama.cpp loads it via GGML_BACKEND_PATH and it registers the wrapper device
add_library(ggml-llamavm-gpu SHARED src/ggml_backend_gpu_shim.cpp)
target_link_libraries(ggml-llamavm-gpu PRIVATE llama_vm)
target_include_directories(ggml-llamavm-gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${LlamaVM_LLAMA_CPP_DIR}/ggml/include
    ${LlamaVM_LLAMA_CPP_DIR}/ggml/src)
set_target_properties(ggml-llamavm-gpu PROPERTIES PREFIX "lib" OUTPUT_NAME "ggml-llamavm-gpu")

# ------------------------------ install ---------------------------------

install(TARGETS llama_vm llama-vm ggml-llamavm-cpu ggml-llamavm-gpu
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
install(FILES src/ggml_buffer.h DESTINATION include/llama_vm)
