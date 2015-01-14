#---------------------------------------------------------------------------------------------------
#  CheckCompiler.cmake
#---------------------------------------------------------------------------------------------------

#---Enable FORTRAN (unfortunatelly is not nowt possible in all cases)-------------------------------
#if(NOT WIN32 AND NOT CMAKE_GENERATOR STREQUAL Xcode AND NOT CMAKE_GENERATOR STREQUAL Ninja)
#  #--Work-around for CMake issue 0009220
#  if(DEFINED CMAKE_Fortran_COMPILER AND CMAKE_Fortran_COMPILER MATCHES "^$")
#    set(CMAKE_Fortran_COMPILER CMAKE_Fortran_COMPILER-NOTFOUND)
#  endif()
#  enable_language(Fortran OPTIONAL)
#endif()


#----Test if clang setup works----------------------------------------------------------------------
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  exec_program(${CMAKE_C_COMPILER} ARGS "-v" OUTPUT_VARIABLE _clang_version_info)
  string(REGEX REPLACE "^.*[ ]([0-9]+)\\.[0-9].*$" "\\1" CLANG_MAJOR "${_clang_version_info}")
  string(REGEX REPLACE "^.*[ ][0-9]+\\.([0-9]).*$" "\\1" CLANG_MINOR "${_clang_version_info}")
else()
  set(CLANG_MAJOR 0)
  set(CLANG_MINOR 0)
endif()

#---Obtain the major and minor version of the GNU compiler-------------------------------------------
if (CMAKE_COMPILER_IS_GNUCXX)
  exec_program(${CMAKE_C_COMPILER} ARGS "-dumpversion" OUTPUT_VARIABLE _gcc_version_info)
  string(REGEX REPLACE "^([0-9]+).*$"                   "\\1" GCC_MAJOR ${_gcc_version_info})
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*$"          "\\1" GCC_MINOR ${_gcc_version_info})
  string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*$" "\\1" GCC_PATCH ${_gcc_version_info})

  if(GCC_PATCH MATCHES "\\.+")
    set(GCC_PATCH "")
  endif()
  if(GCC_MINOR MATCHES "\\.+")
    set(GCC_MINOR "")
  endif()
  if(GCC_MAJOR MATCHES "\\.+")
    set(GCC_MAJOR "")
  endif()
  message(STATUS "Found GCC. Major version ${GCC_MAJOR}, minor version ${GCC_MINOR}")
  set(COMPILER_VERSION gcc${GCC_MAJOR}${GCC_MINOR}${GCC_PATCH})
else()
  set(GCC_MAJOR 0)
  set(GCC_MINOR 0)
endif()
