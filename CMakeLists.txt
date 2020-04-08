# ~~~
# SWIG interface to liquid-dsp
# ~~~

cmake_minimum_required(VERSION 3.12...3.17)

project(
  "liquidswig"
  VERSION 0.1
  DESCRIPTION "Digital Signal Processing library for Software-Defined Radio"
  HOMEPAGE_URL "https://liquidsdr.org/"
  LANGUAGES C CXX)

include(FindPackageHandleStandardArgs)

if(WIN32)
  set(OS_PATH_SEP "\\")
else()
  set(OS_PATH_SEP "/")
endif()

find_package(SWIG)
include(UseSWIG)

find_package(Python COMPONENTS Interpreter Development NumPy)

# Import already compiled libliquid. Yes Windows only, I'll deal with it later
add_library(libliquid SHARED IMPORTED)
set_property(
  TARGET libliquid
  PROPERTY IMPORTED_LOCATION
           "${CMAKE_SOURCE_DIR}/external/liquid/win64gcc/libliquid.dll")
set_property(
  TARGET libliquid
  PROPERTY IMPORTED_IMPLIB
           "${CMAKE_SOURCE_DIR}/external/liquid/win64gcc/libliquid.lib")
set_property(
  TARGET libliquid
  PROPERTY PUBLIC_HEADER "${CMAKE_SOURCE_DIR}/external/liquid/include/liquid.h")

# A library for extra functions that I decide to define
add_library(liquidbindings STATIC "src/liquidbindings.cpp")
target_include_directories(liquidbindings PUBLIC "include/")

# SWIG library to generate bindings
swig_add_library(liquidswigpython LANGUAGE python SOURCES "swig/libliquid.i")
target_link_libraries(liquidswigpython PUBLIC Python::Module Python::NumPy
                                              libliquid liquidbindings)
target_include_directories(
  liquidswigpython
  PRIVATE "${Python_INCLUDE_DIRS}" "include/" "external/liquid/include/"
          "swig/" "${Python_NumPy_INCLUDE_DIRS}")
set_property(
  TARGET liquidswigpython
  PROPERTY SWIG_INCLUDE_DIRECTORIES "${Python_INCLUDE_DIRS}" "include/"
           "external/liquid/include" "swig/" "${Python_NumPy_INCLUDE_DIRS}")
set_property(TARGET liquidswigpython PROPERTY SWIG_COMPILE_OPTIONS "-py3")
set_property(TARGET liquidswigpython PROPERTY SWIG_MODULE_NAME "liquid")

execute_process(
  COMMAND
    ${Python_EXECUTABLE} -c
    "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE)
