# CMakeLists for KEMField/OpenCLPlugin
# T.J. Corona

#This flag is available to set even if the KEMFIELD_USE_OPENCL flag
#is not yet activated. This is  because on certain systems we need to specify
#the platform id before enabling OpenCL so that header file generator
#compiles and runs correctly.
set(${PROJECT_NAME}_OPENCL_PLATFORM "0" CACHE STRING "OpenCL platform ID number.")
mark_as_advanced(FORCE ${PROJECT_NAME}_OPENCL_PLATFORM)
add_cflag(KEMFIELD_OPENCL_PLATFORM=${${PROJECT_NAME}_OPENCL_PLATFORM})

set(${PROJECT_NAME}_OPENCL_DEVICE_TYPE "0" CACHE STRING "OpenCL device type; GPU=0, CPU=1, Accelerator=2.")
mark_as_advanced(FORCE ${PROJECT_NAME}_OPENCL_DEVICE_TYPE)
add_cflag(KEMFIELD_OPENCL_DEVICE_TYPE=${${PROJECT_NAME}_OPENCL_DEVICE_TYPE})

set(${PROJECT_NAME}_OPENCL_FASTDIELECTRICS "0" CACHE STRING "Choose the check method of Neumann boundary elements in Robin Hood; 0 (check interval equal to Dirichlet elements), 1 (decrease checked accuracy of Neumann elements by 1/20), 2 (counter technique with the function RH_BoundaryRatioExceeded)")
mark_as_advanced(FORCE ${PROJECT_NAME}_OPENCL_FASTDIELECTRICS)
add_cflag(KEMFIELD_FASTDIELECTRICS_VALUE=${${PROJECT_NAME}_OPENCL_FASTDIELECTRICS})

cmake_dependent_option (${PROJECT_NAME}_OPENCL_USE_DEPRECATED_API "Use deprecated OpenCL 1.1 API" OFF
  ${PROJECT_NAME}_USE_OPENCL OFF)
mark_as_advanced(FORCE ${PROJECT_NAME}_OPENCL_USE_DEPRECATED_API)
if (${PROJECT_NAME}_OPENCL_USE_DEPRECATED_API)
  set (${PROJECT_NAME}_OPENCL_CFLAGS ${${PROJECT_NAME}_OPENCL_CFLAGS} -DCL_USE_DEPRECATED_OPENCL_1_1_APIS)
  add_cflag (CL_USE_DEPRECATED_OPENCL_1_1_APIS)
endif (${PROJECT_NAME}_OPENCL_USE_DEPRECATED_API)

cmake_dependent_option (${PROJECT_NAME}_OPENCL_USE_CL_VECTOR "Use deprecated cl::vector in place of std::vector" OFF
  ${PROJECT_NAME}_USE_OPENCL OFF)
mark_as_advanced(FORCE ${PROJECT_NAME}_OPENCL_USE_CL_VECTOR)
if (${PROJECT_NAME}_OPENCL_USE_CL_VECTOR)
  set (${PROJECT_NAME}_OPENCL_CFLAGS ${${PROJECT_NAME}_OPENCL_CFLAGS} -DKEMFIELD_USE_CL_VECTOR)
  add_cflag (KEMFIELD_USE_CL_VECTOR)
endif (${PROJECT_NAME}_OPENCL_USE_CL_VECTOR)

cmake_dependent_option (${PROJECT_NAME}_OPENCL_FFT_CONST_MEM "Enable use of OpenCL device constant memory by FFT kernel." ON
  ${PROJECT_NAME}_USE_OPENCL OFF)
mark_as_advanced(FORCE ${PROJECT_NAME}_OPENCL_FFT_CONST_MEM)
if (${PROJECT_NAME}_OPENCL_FFT_CONST_MEM)
  set (${PROJECT_NAME}_OPENCL_CFLAGS ${${PROJECT_NAME}_OPENCL_CFLAGS} -DKEMFIELD_OPENCL_FFT_CONST_MEM)
  add_cflag (KEMFIELD_OPENCL_FFT_CONST_MEM)
endif (${PROJECT_NAME}_OPENCL_FFT_CONST_MEM)

cmake_dependent_option (${PROJECT_NAME}_OPENCL_INTERNAL_1_1 "Use internally packaged OpenCL 1.1 headers" OFF
  ${PROJECT_NAME}_USE_OPENCL OFF)
mark_as_advanced(FORCE ${PROJECT_NAME}_OPENCL_INTERNAL_1_1)
if (${PROJECT_NAME}_OPENCL_INTERNAL_1_1)
  set (${PROJECT_NAME}_OPENCL_TEMP_INCLUDE_DIR_STR \"${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/opencl/1.1\")
  string (REPLACE "\"" "" ${PROJECT_NAME}_OPENCL_TEMP_INCLUDE_DIR_VAR ${${PROJECT_NAME}_OPENCL_TEMP_INCLUDE_DIR_STR} )
  set (${PROJECT_NAME}_OPENCL_CFLAGS ${${PROJECT_NAME}_OPENCL_CFLAGS} -isystem ${${PROJECT_NAME}_OPENCL_TEMP_INCLUDE_DIR_VAR})
endif (${PROJECT_NAME}_OPENCL_INTERNAL_1_1)

cmake_dependent_option (${PROJECT_NAME}_OPENCL_INTERNAL_1_2 "Use internally packaged OpenCL 1.2 headers" OFF
  ${PROJECT_NAME}_USE_OPENCL OFF)
mark_as_advanced(FORCE ${PROJECT_NAME}_OPENCL_INTERNAL_1_2)
if (${PROJECT_NAME}_OPENCL_INTERNAL_1_2)
    set (${PROJECT_NAME}_OPENCL_TEMP_INCLUDE_DIR_STR \"${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/opencl/1.2\")
    string (REPLACE "\"" "" ${PROJECT_NAME}_OPENCL_TEMP_INCLUDE_DIR_VAR ${${PROJECT_NAME}_OPENCL_TEMP_INCLUDE_DIR_STR} )
  set (${PROJECT_NAME}_OPENCL_CFLAGS ${${PROJECT_NAME}_OPENCL_CFLAGS} -isystem ${${PROJECT_NAME}_OPENCL_TEMP_INCLUDE_DIR_VAR})
endif (${PROJECT_NAME}_OPENCL_INTERNAL_1_2)

cmake_dependent_option (${PROJECT_NAME}_OPENCL_FASTRWG "Activate the fast evaluation of electrostatic boundary integrals in RWG basis if no field point has to be computed on any surface." OFF ${PROJECT_NAME}_USE_OPENCL OFF)
mark_as_advanced(FORCE ${PROJECT_NAME}_OPENCL_FASTRWG)
if (${PROJECT_NAME}_OPENCL_FASTRWG)
  add_cflag(KEMFIELD_FASTRWG_VALUE=1)
else (${PROJECT_NAME}_OPENCL_FASTRWG)
  add_cflag(KEMFIELD_FASTRWG_VALUE=0)
endif (${PROJECT_NAME}_OPENCL_FASTRWG)


option (${PROJECT_NAME}_USE_OPENCL "Use OpenCL via C++ wrapper API." OFF)
if (${PROJECT_NAME}_USE_OPENCL)

  # force deactivation of FFTW because it permutes FFT outputs in a different way
  # than the native FFT, which leads to incorrect fast multipole response functions
  set( KEMField_USE_FFTW OFF CACHE BOOL "(Required)" FORCE)

  find_package (OpenCL REQUIRED)

  kasper_external_include_directories(${OPENCL_INCLUDE_DIRS})

  if(${PROJECT_NAME}_OPENCL_INTERNAL_1_1)
    kasper_internal_include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/opencl/1.1)
  endif(${PROJECT_NAME}_OPENCL_INTERNAL_1_1)

  if(${PROJECT_NAME}_OPENCL_INTERNAL_1_2)
    kasper_internal_include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/opencl/1.2)
  endif(${PROJECT_NAME}_OPENCL_INTERNAL_1_2)

  add_cflag (KEMFIELD_USE_OPENCL)

  get_filename_component(OPENCL_LIBDIR ${OPENCL_LIBRARIES} PATH)

  if (APPLE)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER}
      -D KEMFIELD_OPENCL_PLATFORM=${KEMField_OPENCL_PLATFORM}
      -D KEMFIELD_OPENCL_DEVICE_TYPE=${${PROJECT_NAME}_OPENCL_DEVICE_TYPE}
      -o GenerateOpenCLHeader
      -I${OPENCL_INCLUDE_DIRS}
      ${${PROJECT_NAME}_OPENCL_CFLAGS}
      -framework OpenCL
      ${SOURCE}/Plugins/OpenCL/Core/src/GenerateOpenCLHeader.cc
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  else (APPLE)

    execute_process(COMMAND ${CMAKE_CXX_COMPILER}
      -D KEMFIELD_OPENCL_PLATFORM=${${PROJECT_NAME}_OPENCL_PLATFORM}
      -D KEMFIELD_OPENCL_DEVICE_TYPE=${${PROJECT_NAME}_OPENCL_DEVICE_TYPE}
      -o GenerateOpenCLHeader
      -I${OPENCL_INCLUDE_DIRS}
      ${${PROJECT_NAME}_OPENCL_CFLAGS}
      ${SOURCE}/Plugins/OpenCL/Core/src/GenerateOpenCLHeader.cc
      ${OPENCL_LIBRARIES}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  endif (APPLE)

  execute_process(COMMAND ./GenerateOpenCLHeader
    OUTPUT_FILE ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/kEMField_opencl_defines.h
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    RESULT_VARIABLE USE_DOUBLE_PRECISION_GPU)

  if (USE_DOUBLE_PRECISION_GPU STREQUAL "255")
    add_definitions (-DKEMFIELD_DEFAULT_GPU_ID=0)
  else (USE_DOUBLE_PRECISION_GPU STREQUAL "255")
    add_definitions (-DKEMFIELD_USE_DOUBLE_PRECISION)
    add_definitions (-DKEMFIELD_DEFAULT_GPU_ID=${USE_DOUBLE_PRECISION_GPU})
    add_cflag (KEMFIELD_USE_DOUBLE_PRECISION)
    add_cflag(KEMFIELD_DEFAULT_GPU_ID=${USE_DOUBLE_PRECISION_GPU})
  endif (USE_DOUBLE_PRECISION_GPU STREQUAL "255")

#leave this binary around since it is useful for debugging if things go wrong
#execute_process(COMMAND rm GenerateOpenCLHeader WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Core/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/IO/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include)

  set (OPENCLPLUGIN_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KOpenCLHeaderWrapper.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KOpenCLAction.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KOpenCLData.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KOpenCLInterface.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KOpenCLKernelBuilder.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/IO/include/KOpenCLBufferStreamer.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/include/KOpenCLSurfaceContainer.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KOpenCLBoundaryIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KOpenCLBoundaryIntegralMatrix.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KOpenCLBoundaryIntegralVector.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KOpenCLBoundaryIntegralSolutionVector.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/include/KOpenCLElectrostaticBoundaryIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/include/KOpenCLElectrostaticBoundaryIntegratorFactory.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/include/KOpenCLElectrostaticNumericBoundaryIntegrator.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/include/KOpenCLElectrostaticRWGBoundaryIntegrator.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KGaussSeidel_OpenCL.hh
    #${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KFMDenseBlockSparseMatrix_OpenCL.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KRobinHood_OpenCL.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/include/KOpenCLElectrostaticIntegratingFieldSolver.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticMultipoleBatchCalculator_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticFieldMapper_OpenCL.hh
    #${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticBoundaryIntegratorEngine_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticMultipoleCalculator_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticRemoteToRemoteConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticRemoteToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticLocalToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticMultipoleDistributor_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.hh
    #${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticSparseBoundaryIntegralShellMatrix_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticBatchedLocalToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/include/KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMScalarMomentRemoteToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMReducedScalarMomentRemoteToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMScalarMomentLocalToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMScalarMomentRemoteToRemoteConverter_OpenCL.hh
    )

  if (${PROJECT_NAME}_USE_MPI)
    list (APPEND OPENCLPLUGIN_HEADERFILES
      ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KRobinHood_MPI_OpenCL.hh
      #${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticSparseBoundaryIntegralShellMatrix_MPI_OpenCL.hh
      )
  endif (${PROJECT_NAME}_USE_MPI)

  set (OPENCLPLUGIN_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLAction.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLData.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLInterface.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLKernelBuilder.cc
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/src/KOpenCLSurfaceContainer.cc
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/src/KOpenCLElectrostaticBoundaryIntegrator.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/src/KOpenCLElectrostaticBoundaryIntegratorFactory.cc
    #${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/src/KOpenCLElectrostaticNumericBoundaryIntegrator.cc
    #${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/src/KOpenCLElectrostaticRWGBoundaryIntegrator.cc
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticMultipoleBatchCalculator_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticFastMultipoleFieldSolver_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticFieldMapper_OpenCL.cc
    #${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticBoundaryIntegratorEngine_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticMultipoleCalculator_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticRemoteToRemoteConverter_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticRemoteToLocalConverter_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticLocalToLocalConverter_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticMultipoleDistributor_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticBatchedRemoteToLocalConverter_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticBatchedLocalToLocalConverter_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticBatchedRemoteToRemoteConverter_OpenCL.cc
    )

  set_property(
    SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLInterface.cc
    APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_KERNEL_DIR="${${PROJECT_NAME}_DATA_INSTALL_DIR}"
    )

  set (OPENCLPLUGIN_CLFILES
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/kEMField_opencl_defines.h
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/cl/kEMField_GaussianQuadrature.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/cl/kEMField_GaussianCubature.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/cl/kEMField_VectorOperations.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/cl/kEMField_ParallelReduction.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_Rectangle.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_Triangle.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_LineSegment.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_ConicSection.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_SolidAngle.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/cl/kEMField_BoundaryIntegrals.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticRectangle.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticTriangle.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticLineSegment.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticConicSection.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticBoundaryIntegrals.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticBoundaryIntegrals_kernel.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/cl/kEMField_ElectrostaticCubature_CommonFunctions.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/cl/kEMField_ElectrostaticCubatureTriangle_7Point.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/cl/kEMField_ElectrostaticCubatureTriangle_12Point.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/cl/kEMField_ElectrostaticCubatureTriangle_33Point.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/cl/kEMField_ElectrostaticCubatureRectangle_7Point.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/cl/kEMField_ElectrostaticCubatureRectangle_12Point.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/cl/kEMField_ElectrostaticCubatureRectangle_33Point.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/cl/kEMField_ElectrostaticQuadratureLineSegment.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/cl/kEMField_ElectrostaticNumericBoundaryIntegrals.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/cl/kEMField_ElectrostaticNumericBoundaryIntegrals_kernel.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/cl/kEMField_ElectrostaticRWGBoundaryIntegrals.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/cl/kEMField_ElectrostaticRWGBoundaryIntegrals_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/cl/kEMField_ElectrostaticRWGRectangle.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/cl/kEMField_ElectrostaticRWGTriangle.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_LinearAlgebra.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_LinearAlgebra_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_GaussSeidel_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_RobinHood_kernel.cl
    #${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_DenseBlockSparseMatrixVectorProduct_kernel.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/cl/kEMField_ElectrostaticIntegratingFieldSolver_kernel.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMSphericalMultipoleMath.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMMultipoleRotation.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMMultipoleTranslation.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMRotationMatrix.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMArrayMath.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMZeroComplexArray_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMFastFourierTransformUtilities.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMMultidimensionalFastFourierTransform_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMFastFourierTransformUtilitiesPrivate.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMMultidimensionalFastFourierTransformPrivate_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMFastFourierTransformUtilitiesLocal.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMMultidimensionalFastFourierTransformLocal_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMPointwiseComplexVectorAdd_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMComplexMultiply.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticMultipole_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticMultipoleDistribution_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMTriangleMultipole.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMTriangleMultipoleNumerical.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMRectangleMultipole.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMRectangleMultipoleNumerical.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMLineSegmentMultipole.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMLineSegmentMultipoleNumerical.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticRemoteToRemoteCopyAndScale_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticRemoteToRemoteReduceAndScale_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticRemoteToRemoteTransformation_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticRemoteToLocalCopyAndScale_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticRemoteToLocalReduceAndScale_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticLocalToLocalTransformation_kernel.cl
    #${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticSparseShellMatrixVectorProduct_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticBufferedRemoteToLocalCopyAndScale_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticBufferedRemoteToLocalReduceAndScale_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticBatchedLocalToLocalTransformation_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticBatchedRemoteToRemoteReduce_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticBatchedRemoteToRemoteTransformation_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticBatchedRemoteToLocalTransformation_kernel.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentRemoteToRemoteConverter_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentRemoteToLocalConverter_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMReducedScalarMomentRemoteToLocalConverter_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMBufferedReducedScalarMomentRemoteToLocalConverter_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMSparseScalarMomentAdd_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentLocalToLocalConverter_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentApplyScaleFactor_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentArrayReduction_kernel.cl
    )

  add_library (KEMOpenCLPlugin SHARED ${OPENCLPLUGIN_SOURCEFILES})
  target_link_libraries (KEMOpenCLPlugin
    KEMCore
    KEMSurfaces
    KFMKernel
    KFMTree
    KFMMath
    KFMMathUtilities
    KFMElectrostatics
    ${OPENCL_LIBRARIES}
    )

  kasper_install_headers (${OPENCLPLUGIN_HEADERFILES})
  kasper_install_libraries (KEMOpenCLPlugin)

  # Generate GPU-specific headers for OpenCL implementation of KEMField

  install ( FILES ${OPENCLPLUGIN_CLFILES}
    DESTINATION ${${PROJECT_NAME}_DATA_INSTALL_DIR}
    )

endif (${PROJECT_NAME}_USE_OPENCL)
