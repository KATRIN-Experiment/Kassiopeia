# CMakeLists for KEMField/OpenCLPlugin
# T.J. Corona

cmake_dependent_option (@PROJECT_NAME@_OPENCL_CPU "Use OpenCL on the CPU" OFF
  @PROJECT_NAME@_USE_OPENCL OFF)
if (@PROJECT_NAME@_OPENCL_CPU)
  set (@PROJECT_NAME@_OPENCL_CFLAGS -DKEMFIELD_OPENCL_CPU)
  add_cflag (KEMFIELD_OPENCL_CPU)
endif (@PROJECT_NAME@_OPENCL_CPU)

cmake_dependent_option (@PROJECT_NAME@_OPENCL_ANALYTIC_MULTIPOLE "Use OpenCL to compute the element multipole moments analytically." OFF
  @PROJECT_NAME@_USE_OPENCL OFF)
if (@PROJECT_NAME@_OPENCL_ANALYTIC_MULTIPOLE)
  set (@PROJECT_NAME@_OPENCL_CFLAGS -DKEMFIELD_OPENCL_ANALYTIC_MULTIPOLE)
  add_cflag (KEMFIELD_OPENCL_ANALYTIC_MULTIPOLE)
endif (@PROJECT_NAME@_OPENCL_ANALYTIC_MULTIPOLE)

option (@PROJECT_NAME@_USE_OPENCL "Use OpenCL" OFF)
if (@PROJECT_NAME@_USE_OPENCL)
  find_package (OpenCL REQUIRED)
  kasper_external_include_directories(${OPENCL_INCLUDE_DIRS})
  kasper_external_include_directories(${CMAKE_SOURCE_DIR}/@PROJECT_NAME@/opencl/1.1)
  add_cflag (KEMFIELD_USE_OPENCL)

  if (APPLE)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER}
      -o GenerateOpenCLHeader
      -I${OPENCL_INCLUDE_DIRS}
      -isystem${CMAKE_SOURCE_DIR}/@PROJECT_NAME@/opencl/1.1
      ${@PROJECT_NAME@_OPENCL_CFLAGS}
      -framework OpenCL
      ${SOURCE}/Plugins/OpenCL/Core/src/GenerateOpenCLHeader.cc
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  else (APPLE)
    get_filename_component(OPENCL_LIBDIR ${OPENCL_LIBRARIES} PATH)
    execute_process(COMMAND ${CMAKE_CXX_COMPILER}
      -o GenerateOpenCLHeader
      -I${OPENCL_INCLUDE_DIRS}
      -isystem${CMAKE_SOURCE_DIR}/@PROJECT_NAME@/opencl/1.1
      ${@PROJECT_NAME@_OPENCL_CFLAGS}
      ${SOURCE}/Plugins/OpenCL/Core/src/GenerateOpenCLHeader.cc
      -L${OPENCL_LIBDIR} -lOpenCL
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  endif (APPLE)

  execute_process(COMMAND ./GenerateOpenCLHeader
    OUTPUT_FILE ${CMAKE_BINARY_DIR}/@PROJECT_NAME@/kEMField_defines.h
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

  execute_process(COMMAND rm GenerateOpenCLHeader
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Core/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/IO/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/include)
  kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include)

  set (OPENCLPLUGIN_HEADERFILES
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
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KGaussSeidel_OpenCL.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KRobinHood_OpenCL.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/include/KOpenCLElectrostaticIntegratingFieldSolver.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticMultipoleBatchCalculator_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticFastMultipoleFieldSolver_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticFieldMapper_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticBoundaryIntegratorEngine_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticMultipoleCalculator_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticRemoteToRemoteConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticMultipoleDistributor_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.hh
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/include/KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMScalarMomentRemoteToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMReducedScalarMomentRemoteToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMScalarMomentLocalToLocalConverter_OpenCL.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMScalarMomentRemoteToRemoteConverter_OpenCL.hh
    )

  if (@PROJECT_NAME@_USE_MPI)
    list (APPEND OPENCLPLUGIN_HEADERFILES
      ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KRobinHood_MPI_OpenCL.hh
      )
  endif (@PROJECT_NAME@_USE_MPI)

  set (OPENCLPLUGIN_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLAction.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLData.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLInterface.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLKernelBuilder.cc
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/src/KOpenCLSurfaceContainer.cc
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/src/KOpenCLElectrostaticBoundaryIntegrator.cc
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/src/KOpenCLElectrostaticIntegratingFieldSolver.cc
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticMultipoleBatchCalculator_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticFastMultipoleFieldSolver_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticFieldMapper_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticBoundaryIntegratorEngine_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticMultipoleCalculator_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticRemoteToRemoteConverter_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMElectrostaticMultipoleDistributor_OpenCL.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/src/KFMSparseElectrostaticBoundaryIntegratorEngine_OpenCL.cc
    )

  set_property(
    SOURCE
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLInterface.cc
    APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_KERNEL_DIR="${@PROJECT_NAME@_DATA_INSTALL_DIR}"
    )

  set (OPENCLPLUGIN_CLFILES
    ${CMAKE_BINARY_DIR}/@PROJECT_NAME@/kEMField_defines.h
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Math/cl/kEMField_GaussianQuadrature.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Core/cl/kEMField_ParallelReduction.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_Rectangle.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_Triangle.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_LineSegment.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_ConicSection.cl
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
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_LinearAlgebra.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_LinearAlgebra_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_GaussSeidel_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_RobinHood_kernel.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/cl/kEMField_ElectrostaticIntegratingFieldSolver_kernel.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMSphericalMultipoleMath.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMMultipoleRotation.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMMultipoleTranslation.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMRotationMatrix.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMArrayMath.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMFastFourierTransformUtilities.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMMultidimensionalFastFourierTransform_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/cl/kEMField_KFMPointwiseComplexVectorAdd_kernel.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticMultipole_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticMultipoleDistribution_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMTriangleMultipole.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMRectangleMultipole.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMLineSegmentMultipole.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticRemoteToRemoteCopyAndScale_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticRemoteToRemoteReduceAndScale_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/cl/kEMField_KFMElectrostaticRemoteToRemoteTransformation_kernel.cl
    #
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentRemoteToRemoteConverter_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentRemoteToLocalConverter_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMReducedScalarMomentRemoteToLocalConverter_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMSparseScalarMomentAdd_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentLocalToLocalConverter_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentApplyScaleFactor_kernel.cl
    ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentArrayReduction_kernel.cl
    )

  add_library (KEMOpenCLPlugin SHARED ${OPENCLPLUGIN_SOURCEFILES})
  target_link_libraries (KEMOpenCLPlugin KEMCore KEMSurfaces KEMIntegratingFieldSolver KFMKernel KFMTree KFMMath KFMMathUtilities KFMElectrostatics)

  kasper_install_headers (${OPENCLPLUGIN_HEADERFILES})
  kasper_install_libraries (KEMOpenCLPlugin)

  # Generate GPU-specific headers for OpenCL implementation of KEMField

  install ( FILES ${OPENCLPLUGIN_CLFILES}
    DESTINATION ${@PROJECT_NAME@_DATA_INSTALL_DIR}
    )

endif (@PROJECT_NAME@_USE_OPENCL)
