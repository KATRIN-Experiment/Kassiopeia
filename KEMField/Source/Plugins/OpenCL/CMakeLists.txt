# CMakeLists for KEMField/OpenCLPlugin
# T.J. Corona

#This flag is available to set even if the KEMFIELD_USE_OPENCL flag
#is not yet activated. This is  because on certain systems we need to specify
#the platform id before enabling OpenCL so that header file generator
#compiles and runs correctly.

set(KEMField_OPENCL_PLATFORM "0" CACHE STRING "OpenCL platform ID number.")
mark_as_advanced(FORCE KEMField_OPENCL_PLATFORM)

set(KEMField_OPENCL_DEVICE_TYPE "0" CACHE STRING "OpenCL device type; GPU=0, CPU=1, Accelerator=2; Any=-1.")
mark_as_advanced(FORCE KEMField_OPENCL_DEVICE_TYPE)

set (KEMField_OPENCL_FASTDIELECTRICS "0" CACHE STRING
    "Choose the check method of Neumann boundary elements in Robin Hood; 0 (check interval equal to Dirichlet elements), 1 (decrease checked accuracy of Neumann elements by 1/20), 2 (counter technique with the function RH_BoundaryRatioExceeded)")
mark_as_advanced(FORCE KEMField_OPENCL_FASTDIELECTRICS)

cmake_dependent_option (KEMField_OPENCL_FASTRWG
    "Activate the fast evaluation of electrostatic boundary integrals in RWG basis if no field point has to be computed on any surface."
    OFF KEMField_USE_OPENCL OFF)
mark_as_advanced(FORCE KEMField_OPENCL_FASTRWG)

cmake_dependent_option (KEMField_OPENCL_FFT_CONST_MEM "Enable use of OpenCL device constant memory by FFT kernel." ON KEMField_USE_OPENCL OFF)
mark_as_advanced(FORCE KEMField_OPENCL_FFT_CONST_MEM)
if (KEMField_OPENCL_FFT_CONST_MEM)
    set (KEMField_OPENCL_CFLAGS ${KEMField_OPENCL_CFLAGS} -DKEMFIELD_OPENCL_FFT_CONST_MEM)
endif (KEMField_OPENCL_FFT_CONST_MEM)

cmake_dependent_option (KEMField_OPENCL_USE_CL_VECTOR "Use deprecated cl::vector in place of std::vector" OFF KEMField_USE_OPENCL OFF)
mark_as_advanced(FORCE KEMField_OPENCL_USE_CL_VECTOR)
if (KEMField_OPENCL_USE_CL_VECTOR)
    set (KEMField_OPENCL_CFLAGS ${KEMField_OPENCL_CFLAGS} -DKEMFIELD_USE_CL_VECTOR)
endif (KEMField_OPENCL_USE_CL_VECTOR)

cmake_dependent_option (KEMField_OPENCL_INTERNAL_2_2 "Use internally packaged OpenCL 2.2 headers" OFF KEMField_USE_OPENCL OFF)
mark_as_advanced(FORCE KEMField_OPENCL_INTERNAL_2_2)
if (KEMField_OPENCL_INTERNAL_2_2)
    set (KEMField_OPENCL_TEMP_INCLUDE_DIR_STR \"${CMAKE_SOURCE_DIR}/${PROJECT_NAME}/opencl/2.2\")
    string (REPLACE "\"" "" KEMField_OPENCL_TEMP_INCLUDE_DIR_VAR ${KEMField_OPENCL_TEMP_INCLUDE_DIR_STR} )
    set (KEMField_OPENCL_CFLAGS ${KEMField_OPENCL_CFLAGS} -isystem ${KEMField_OPENCL_TEMP_INCLUDE_DIR_VAR})
endif (KEMField_OPENCL_INTERNAL_2_2)

option (KEMField_USE_OPENCL "Use OpenCL via C++ wrapper API." OFF)
if (KEMField_USE_OPENCL)

    # force deactivation of FFTW because it permutes FFT outputs in a different way
    # than the native FFT, which leads to incorrect fast multipole response functions
    set( KEMField_USE_FFTW OFF CACHE BOOL "(Required)" FORCE)

    find_package (OpenCL REQUIRED)

    if(KEMField_OPENCL_INTERNAL_2_2)
        set(OpenCL_INCLUDE_DIRS SYSTEM ${KEMField_OPENCL_TEMP_INCLUDE_DIR_STR})
    endif(KEMField_OPENCL_INTERNAL_2_2)

    message(STATUS "${PROJECT_NAME} is looking for compatbile OpenCL devices ...")

    if (APPLE)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER}
            -D KEMFIELD_OPENCL_PLATFORM=${KEMField_OPENCL_PLATFORM}
            -D KEMFIELD_OPENCL_DEVICE_TYPE=${KEMField_OPENCL_DEVICE_TYPE}
            -o ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/GenerateOpenCLHeader
            -I${OpenCL_INCLUDE_DIRS}
            ${KEMField_OPENCL_CFLAGS}
            -framework OpenCL
            ${SOURCE}/Plugins/OpenCL/Core/src/GenerateOpenCLHeader.cc
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            RESULT_VARIABLE COMPILE_STATUS
        )
    else (APPLE)
        execute_process(COMMAND ${CMAKE_CXX_COMPILER}
            -D KEMFIELD_OPENCL_PLATFORM=${KEMField_OPENCL_PLATFORM}
            -D KEMFIELD_OPENCL_DEVICE_TYPE=${KEMField_OPENCL_DEVICE_TYPE}
            -o ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/GenerateOpenCLHeader
            -I${OpenCL_INCLUDE_DIRS}
            ${KEMField_OPENCL_CFLAGS}
            ${SOURCE}/Plugins/OpenCL/Core/src/GenerateOpenCLHeader.cc
            ${OpenCL_LIBRARIES}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            RESULT_VARIABLE COMPILE_STATUS
        )
    endif (APPLE)

    if(COMPILE_STATUS AND NOT COMPILE_STATUS EQUAL 0)
        message(FATAL_ERROR " *** Could not build \"GenerateOpenCLHeader\" ***\n"
            " There was an error compiling an OpenCL program. Please check your config.\n"
            "   KEMFIELD_OPENCL_PLATFORM=${KEMField_OPENCL_PLATFORM}\n"
            "   KEMFIELD_OPENCL_DEVICE_TYPE=${KEMField_OPENCL_DEVICE_TYPE}")
    endif()

    set(OPENCLPLUGIN_GENERATED_HEADER  ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/kEMField_opencl_defines.h)
    execute_process(COMMAND ${CMAKE_BINARY_DIR}/${PROJECT_NAME}/GenerateOpenCLHeader
        OUTPUT_FILE ${OPENCLPLUGIN_GENERATED_HEADER}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        RESULT_VARIABLE EXEC_STATUS
    )

    if(EXEC_STATUS AND NOT EXEC_STATUS EQUAL 0)
        file(REMOVE ${OPENCLPLUGIN_GENERATED_HEADER})
        message(FATAL_ERROR " *** Could not run \"GenerateOpenCLHeader\"  ***\n"
            " There was an error generating the OpenCL headers. Please check your config.\n"
            "   KEMFIELD_OPENCL_PLATFORM=${KEMField_OPENCL_PLATFORM}\n"
            "   KEMFIELD_OPENCL_DEVICE_TYPE=${KEMField_OPENCL_DEVICE_TYPE}")
    else()
        message(STATUS "OpenCL header was generated: ${OPENCLPLUGIN_GENERATED_HEADER}")
    endif()

    #leave this binary around since it is useful for debugging if things go wrong
    #execute_process(COMMAND rm GenerateOpenCLHeader WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    set (OPENCLPLUGIN_HEADERFILES
        ${OPENCLPLUGIN_GENERATED_HEADER}

        ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KOpenCLHeaderWrapper.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KOpenCLAction.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KOpenCLData.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KOpenCLInterface.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/Core/include/KOpenCLKernelBuilder.hh

        ${CMAKE_CURRENT_SOURCE_DIR}/IO/include/KOpenCLBufferStreamer.hh

        ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/include/KOpenCLSurfaceContainer.hh

        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KOpenCLBoundaryIntegrator.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KOpenCLBoundaryIntegralMatrix.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KOpenCLBoundaryIntegralVector.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/include/KOpenCLBoundaryIntegralSolutionVector.hh

        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/include/KOpenCLElectrostaticBoundaryIntegrator.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/include/KOpenCLElectrostaticBoundaryIntegratorConfig.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/include/KOpenCLElectrostaticBoundaryIntegratorFactory.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/include/KOpenCLElectrostaticNumericBoundaryIntegrator.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/include/KOpenCLElectrostaticRWGBoundaryIntegrator.hh

        ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KGaussSeidel_OpenCL.hh
        #${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KFMDenseBlockSparseMatrix_OpenCL.hh

        ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KRobinHood_OpenCL.hh

        ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/include/KOpenCLElectrostaticIntegratingFieldSolver.hh

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

        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Math/include/KFMBatchedMultidimensionalFastFourierTransform_OpenCL.hh

        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMScalarMomentRemoteToLocalConverter_OpenCL.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMReducedScalarMomentRemoteToLocalConverter_OpenCL.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMSparseReducedScalarMomentRemoteToLocalConverter_OpenCL.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMScalarMomentLocalToLocalConverter_OpenCL.hh
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/include/KFMScalarMomentRemoteToRemoteConverter_OpenCL.hh
    )

    if (KEMField_USE_MPI)
        list (APPEND OPENCLPLUGIN_HEADERFILES
            ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/include/KRobinHood_MPI_OpenCL.hh
            #${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Electrostatics/include/KFMElectrostaticSparseBoundaryIntegralShellMatrix_MPI_OpenCL.hh
        )
    endif (KEMField_USE_MPI)

    set (OPENCLPLUGIN_SOURCEFILES
        ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLAction.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLData.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLInterface.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLKernelBuilder.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/src/KOpenCLSurfaceContainer.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/src/KOpenCLElectrostaticBoundaryIntegrator.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/src/KOpenCLElectrostaticBoundaryIntegratorFactory.cc
        #${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/ElectrostaticNumeric/src/KOpenCLElectrostaticNumericBoundaryIntegrator.cc
        #${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/src/KOpenCLElectrostaticRWGBoundaryIntegrator.cc

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

    set_property(SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/Core/src/KOpenCLInterface.cc
        APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_KERNEL_DIR="${KEMField_DATA_INSTALL_DIR}")

    set (OPENCLPLUGIN_CLFILES
        ${OPENCLPLUGIN_GENERATED_HEADER}

        ${CMAKE_CURRENT_SOURCE_DIR}/Math/cl/kEMField_GaussianQuadrature.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/Math/cl/kEMField_GaussianCubature.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/Math/cl/kEMField_VectorOperations.cl

        ${CMAKE_CURRENT_SOURCE_DIR}/Core/cl/kEMField_ParallelReduction.cl

        ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_Rectangle.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_Triangle.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_LineSegment.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_ConicSection.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/Surfaces/cl/kEMField_SolidAngle.cl

        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Core/cl/kEMField_BoundaryIntegrals.cl

        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticRectangle.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticTriangle.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticLineSegment.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticConicSection.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticBoundaryIntegrals.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/Electrostatic/cl/kEMField_ElectrostaticBoundaryIntegrals_kernel.cl

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

        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/cl/kEMField_ElectrostaticRWGBoundaryIntegrals.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/cl/kEMField_ElectrostaticRWGBoundaryIntegrals_kernel.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/cl/kEMField_ElectrostaticRWGRectangle.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/BoundaryIntegrals/RWG/cl/kEMField_ElectrostaticRWGTriangle.cl

        ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_LinearAlgebra.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_LinearAlgebra_kernel.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_GaussSeidel_kernel.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_RobinHood_kernel.cl
        #${CMAKE_CURRENT_SOURCE_DIR}/LinearAlgebra/cl/kEMField_DenseBlockSparseMatrixVectorProduct_kernel.cl

        ${CMAKE_CURRENT_SOURCE_DIR}/FieldSolvers/Integrating/cl/kEMField_ElectrostaticIntegratingFieldSolver_kernel.cl

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

        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentRemoteToRemoteConverter_kernel.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentRemoteToLocalConverter_kernel.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMReducedScalarMomentRemoteToLocalConverter_kernel.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMBufferedReducedScalarMomentRemoteToLocalConverter_kernel.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMSparseScalarMomentAdd_kernel.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentLocalToLocalConverter_kernel.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentApplyScaleFactor_kernel.cl
        ${CMAKE_CURRENT_SOURCE_DIR}/FastMultipole/Tree/cl/kEMField_KFMScalarMomentArrayReduction_kernel.cl
    )

    add_library (KEMOpenCLPlugin SHARED
        ${OPENCLPLUGIN_SOURCEFILES} ${OPENCLPLUGIN_HEADERFILES})

    # get header paths from collected header files
    foreach(HEADER ${OPENCLPLUGIN_HEADERFILES})
        get_filename_component(DIRNAME ${HEADER} DIRECTORY)
        target_include_directories(KEMOpenCLPlugin PUBLIC $<BUILD_INTERFACE:${DIRNAME}>)
    endforeach(HEADER)
    target_include_directories(KEMOpenCLPlugin PUBLIC $<INSTALL_INTERFACE:include>)

    target_compile_definitions(KEMOpenCLPlugin
        PUBLIC
            KEMFIELD_USE_OPENCL
            KEMFIELD_OPENCL_PLATFORM=${KEMField_OPENCL_PLATFORM}
            KEMFIELD_OPENCL_DEVICE_TYPE=${KEMField_OPENCL_DEVICE_TYPE}
    )

    target_link_libraries (KEMOpenCLPlugin
        PUBLIC
            KEMCore
            KEMSurfaces
            KEMElectrostaticBoundaryIntegrals
            KEMIntegratingSolver
            KFMElectrostatics
            OpenCL::OpenCL
    )

    if(KEMField_OPENCL_FASTRWG)
        target_compile_definitions(KEMOpenCLPlugin PRIVATE KEMFIELD_FASTRWG_VALUE=1)
    else(KEMField_OPENCL_FASTRWG)
        target_compile_definitions(KEMOpenCLPlugin PRIVATE KEMFIELD_FASTRWG_VALUE=0)
    endif(KEMField_OPENCL_FASTRWG)
    if(KEMField_OPENCL_FFT_CONST_MEM)
        target_compile_definitions(KEMOpenCLPlugin PRIVATE KEMFIELD_OPENCL_FFT_CONST_MEM)
    endif(KEMField_OPENCL_FFT_CONST_MEM)
    if(KEMField_OPENCL_USE_CL_VECTOR)
        target_compile_definitions(KEMOpenCLPlugin PRIVATE KEMFIELD_USE_CL_VECTOR)
    endif(KEMField_OPENCL_USE_CL_VECTOR)

    if( CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_COMPILER_IS_GNUCXX )
        if( NOT CMAKE_CXX_COMPILER_VERSION VERSION_LESS "7.0" )
            # Use -faligned-new flag to trick gcc into using aligned allocators CL types.
            # This is only used by KEMField, but required here because of include files.
            target_compile_options(KEMOpenCLPlugin PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-faligned-new> )
        endif()
    endif()

    kasper_install_headers (${OPENCLPLUGIN_HEADERFILES})
    kasper_install_libraries (KEMOpenCLPlugin)

    # Generate GPU-specific headers for OpenCL implementation of KEMField

    install ( FILES ${OPENCLPLUGIN_CLFILES} DESTINATION ${KEMField_DATA_INSTALL_DIR} )

endif (KEMField_USE_OPENCL)
