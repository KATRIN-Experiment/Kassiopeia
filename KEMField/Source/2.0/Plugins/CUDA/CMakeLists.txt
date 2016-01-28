# CMakeLists for KEMField/CUDAPlugin
# Daniel Hilk

option (@PROJECT_NAME@_USE_CUDA "Use CUDA via Runtime API, C++11 enabled by default, requires OpenCL and VTK to be deactivated." OFF)

if (@PROJECT_NAME@_USE_CUDA)

  # deactivate OpenCL because linking against two APIs can lead to program compilation problems
  set( KEMField_USE_OPENCL OFF CACHE BOOL "(Required)" FORCE)

  # deactivate VTK, because nvcc doesn't like VTK version 6+ implementation module compile definitions 
  set( KEMField_USE_VTK OFF CACHE BOOL "(Required)" FORCE)

  FIND_PACKAGE(CUDA REQUIRED)
  INCLUDE(FindCUDA)


  # option in order to explicitly de-/activate double precision (fp64)

  cmake_dependent_option( CUDA_USE_FP64 "De-/activate double precision (fp64)." ON
  @PROJECT_NAME@_USE_CUDA OFF)
  mark_as_advanced( CUDA_USE_FP64 )
  if (CUDA_USE_FP64)
    add_definitions (-DKEMFIELD_USE_DOUBLE_PRECISION)
    add_cflag (KEMFIELD_USE_DOUBLE_PRECISION)
  endif (CUDA_USE_FP64)


  # CUDA Fermi architecture

  cmake_dependent_option( CUDA_USE_FERMI_ARCHITECTURE "Compile the CUDA code for compute capability version 2.0 (Fermi chips)." ON
  @PROJECT_NAME@_USE_CUDA OFF)
  if (CUDA_USE_FERMI_ARCHITECTURE)
    set( CUDA_USE_KEPLER_ARCHITECTURE OFF CACHE BOOL "Compile the CUDA code for compute capability version 3.5 (Kepler chips)." FORCE )
    set( CUDA_USE_INDIVIDUAL_ARCHITECTURE OFF CACHE BOOL "Compile the CUDA code for user-defined compute capability version." FORCE )
    set( CUDA_NVCC_FLAGS "-arch=compute_20 -code=sm_20 -std=c++11" CACHE STRING "Compile the CUDA code for compute capability version 2.0 (Fermi chips).BLA" FORCE )
  endif (CUDA_USE_FERMI_ARCHITECTURE)


  # CUDA Kepler architecture
 
 cmake_dependent_option( CUDA_USE_KEPLER_ARCHITECTURE "Compile the CUDA code for compute capability version 3.5 (Kepler chips)." OFF
  @PROJECT_NAME@_USE_CUDA OFF)
  if (CUDA_USE_KEPLER_ARCHITECTURE)
    set( CUDA_USE_FERMI_ARCHITECTURE OFF CACHE BOOL "Compile the CUDA code for compute capability version 2.0 (Fermi chips)." FORCE )
    set( CUDA_USE_INDIVIDUAL_ARCHITECTURE OFF CACHE BOOL "Compile the CUDA code for user-defined compute capability version." FORCE )
    set( CUDA_NVCC_FLAGS "-arch=compute_35 -code=sm_35 -std=c++11" CACHE STRING "Compile the CUDA code for compute capability version 3.5 (Kepler chips).BLA" FORCE )
  endif (CUDA_USE_KEPLER_ARCHITECTURE)


  # user-defined CUDA compute capability

  cmake_dependent_option( CUDA_USE_INDIVIDUAL_ARCHITECTURE "Compile the CUDA code for user-defined compute capability version." OFF
  @PROJECT_NAME@_USE_CUDA OFF)
  if (CUDA_USE_INDIVIDUAL_ARCHITECTURE)
    set( CUDA_USE_FERMI_ARCHITECTURE OFF CACHE BOOL "Compile the CUDA code for compute capability version 2.0 (Fermi chips)." FORCE )
    set( CUDA_USE_KEPLER_ARCHITECTURE OFF CACHE BOOL "Compile the CUDA code for compute capability version 3.5 (Kepler chips)." FORCE )

    set( CUDA_CC_MAJOR_VERSION "5" CACHE STRING "Major compute capability version." )
    set( CUDA_CC_MINOR_VERSION "2" CACHE STRING "Minor compute capability version." )

    set( CUDA_NVCC_FLAGS "-arch=compute_${CUDA_CC_MAJOR_VERSION}${CUDA_CC_MINOR_VERSION} -code=sm_${CUDA_CC_MAJOR_VERSION}${CUDA_CC_MINOR_VERSION} -std=c++11" CACHE STRING "Compile the CUDA code for user-defined compute capability version." FORCE )
  endif (CUDA_USE_INDIVIDUAL_ARCHITECTURE)


  # selected CUDA device

  set( CUDA_DEVICE_ID "0" CACHE STRING "Default CUDA device ID." )
  mark_as_advanced( CUDA_DEVICE_ID )
  add_definitions ( -DKEMFIELD_DEFAULT_GPU_ID=${CUDA_DEVICE_ID} )
  add_cflag( KEMFIELD_DEFAULT_GPU_ID=${CUDA_DEVICE_ID} )


  # get thread block sizes suggested by CUDA Occupancy API (needs CUDA 6.5)

  set( @PROJECT_NAME@_CUDABLOCKSIZES "0" CACHE STRING "View thread block sizes suggested by CUDA Occupancy API (needs CUDA 6.5)." )
  mark_as_advanced( @PROJECT_NAME@_CUDABLOCKSIZES )
  add_definitions ( -DKEMFIELD_OCCUPANCYAPI=${@PROJECT_NAME@_CUDABLOCKSIZES} )
  add_cflag( KEMFIELD_OCCUPANCYAPI=${@PROJECT_NAME@_CUDABLOCKSIZES} )


  # determine path of gcc via 'which' command (not from environment variable 'CC'!)

  execute_process(COMMAND which gcc
  OUTPUT_VARIABLE NVCC_COMPILER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(CUDA_HOST_COMPILER ${NVCC_COMPILER} CACHE STRING "NVCC works on linux only with the GCC compiler correctly." FORCE)


  kasper_external_include_directories(${CUDA_INCLUDE_DIRS})
  add_cflag (KEMFIELD_USE_CUDA)
  get_filename_component(CUDA_LIBDIR ${CUDA_LIBRARIES} PATH)


  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/Core/include )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/Core/cu )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/IO/include )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/include )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cu )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/Math/cu )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/cu )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/include )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cu )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cu )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/include )
  kasper_internal_include_directories( ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/cu )
  

  set (CUDAPLUGIN_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/kEMField_cuda_defines.h
    #    
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KCUDAAction.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KCUDAData.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KCUDAInterface.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/IO/include/KCUDABufferStreamer.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/include/KCUDASurfaceContainer.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KCUDABoundaryIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KCUDABoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KCUDABoundaryIntegralVector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KCUDABoundaryIntegralSolutionVector.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/include/KCUDAElectrostaticBoundaryIntegrator.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KRobinHood_CUDA.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/include/KCUDAElectrostaticIntegratingFieldSolver.hh
  )


  set (CUDAPLUGIN_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KCUDAAction.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KCUDAData.cu
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KCUDAInterface.cu
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/src/KCUDASurfaceContainer.cu
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/src/KCUDAElectrostaticBoundaryIntegrator.cu
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/src/KCUDAElectrostaticIntegratingFieldSolver.cu
  )


  cuda_add_library( KEMCUDAPlugin SHARED ${CUDAPLUGIN_SOURCEFILES} )
  target_link_libraries (KEMCUDAPlugin
    KEMCore
    KEMSurfaces
    KEMIntegratingFieldSolver
    ${CUDA_LIBRARIES}
  )


  kasper_install_headers (${CUDAPLUGIN_HEADERFILES})
  kasper_install_libraries (KEMCUDAPlugin)


endif (@PROJECT_NAME@_USE_CUDA)
