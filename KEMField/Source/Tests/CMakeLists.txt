# CMakeLists for KEMField/Tests
# Author: T.J. Corona

option (@PROJECT_NAME@_ENABLE_TEST "Build test applications" OFF)
if (@PROJECT_NAME@_ENABLE_TEST)
  enable_testing ()
endif (@PROJECT_NAME@_ENABLE_TEST)

if (@PROJECT_NAME@_ENABLE_TEST)

    kasper_internal_include_directories(${@PROJECT_NAME@_INCLUDE_DIRS})
  
  set (TESTS_HEADERS
    include/KElectrostaticBoundaryIntegratorOptions.hh
    include/TestCubatureTriangleIntegrator.hh
    include/TestRWGTriangleIntegrator.hh
    )

    kasper_internal_include_directories( include )
    kasper_install_headers( ${TESTS_HEADERS} )

  set (TESTS_LIBS
    ${ROOT_LIBRARIES}
    ${GSL_LIBRARIES}
    KEMCore
    KEMHashGenerator
    KEMStructuredASCII
    KEMMath
    KFMMathUtilities
    KEMSurfaces
    KEMIO
    KEMFileManipulation
    KEMElectrostaticBoundaryIntegrals
    KEMElectromagnets
    KEMZHGenerator
    KEMZHSolver
    KEMVisualization
    )

  if (@PROJECT_NAME@_USE_VTK)
    list (APPEND TESTS_LIBS
    ${Kommon_Vtk_LIBRARIES}
    KEMVTKPlugin
    )
  endif (@PROJECT_NAME@_USE_VTK)

  if (@PROJECT_NAME@_USE_ROOT)
    find_package(ROOT REQUIRED)
    kasper_external_include_directories( ${ROOT_INCLUDE_DIRS} )
    list (APPEND TESTS_LIBS
    KEMRootPlugin
    )
  endif (@PROJECT_NAME@_USE_ROOT)

  if (@PROJECT_NAME@_USE_OPENCL)
    find_package(OpenCL REQUIRED)
    kasper_external_include_directories( ${OPENCL_INCLUDE_DIRS} )
    list (APPEND TESTS_LIBS
    KEMOpenCLPlugin
    )
  endif (@PROJECT_NAME@_USE_OPENCL)

  if (@PROJECT_NAME@_USE_KGEOBAG)
    kasper_find_module(KGeoBag)
    list (APPEND TESTS_LIBS
      KEMGeoBagPlugin
      ${KOMMON_LIBRARIES}
    )
  endif (@PROJECT_NAME@_USE_KGEOBAG)

  if (@PROJECT_NAME@_USE_KMESSAGE)
    kasper_find_module(Kommon)
    list (APPEND TESTS_LIBS
      ${KOMMON_LIBRARIES}
    )
  endif (@PROJECT_NAME@_USE_KMESSAGE)

  if (@PROJECT_NAME@_USE_PETSc)
    list (APPEND TESTS_LIBS
    KEMPETScPlugin
    )
  endif (@PROJECT_NAME@_USE_PETSc)

  if (@PROJECT_NAME@_USE_VTK)
    list (APPEND TESTS_LIBS
      ${VTK_LIBRARIES})
  endif (@PROJECT_NAME@_USE_VTK)

  add_executable (TestElectromagnets
    ${CMAKE_CURRENT_SOURCE_DIR}/TestElectromagnets.cc)
  target_link_libraries (TestElectromagnets ${TESTS_LIBS} )

  add_executable (TestElectromagnetViewer
    ${CMAKE_CURRENT_SOURCE_DIR}/TestElectromagnetViewer.cc)
  target_link_libraries (TestElectromagnetViewer ${TESTS_LIBS} )

  add_executable (TestDiskCapacitance
    ${CMAKE_CURRENT_SOURCE_DIR}/TestDiskCapacitance.cc)
  target_link_libraries (TestDiskCapacitance ${TESTS_LIBS} )

  add_executable (TestEMFileInterface
    ${CMAKE_CURRENT_SOURCE_DIR}/TestEMFileInterface.cc)
  target_link_libraries (TestEMFileInterface ${TESTS_LIBS} )

  add_executable (TestGeometry
    ${CMAKE_CURRENT_SOURCE_DIR}/TestGeometry.cc)
  target_link_libraries (TestGeometry ${TESTS_LIBS} )

  add_executable (TestAnisotropicPoisson
    ${CMAKE_CURRENT_SOURCE_DIR}/TestAnisotropicPoisson.cc)
  target_link_libraries (TestAnisotropicPoisson ${TESTS_LIBS} )
  
  add_executable (TestCubatureSpeedStandAlone
    ${CMAKE_CURRENT_SOURCE_DIR}/TestCubatureSpeedStandAlone.cc)
  target_link_libraries (TestCubatureSpeedStandAlone ${TESTS_LIBS} )
  
  add_executable (TestIntegratorFunctionality
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorFunctionality.cc)
  target_link_libraries (TestIntegratorFunctionality ${TESTS_LIBS} )
  
  add_executable (TestIntegratorSpeed
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorSpeed.cc)
  target_link_libraries (TestIntegratorSpeed ${TESTS_LIBS} )

  add_executable (TestIntegratorSpeedWithKbdInput
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorSpeedWithKbdInput.cc)
  target_link_libraries (TestIntegratorSpeedWithKbdInput ${TESTS_LIBS} )

  add_executable (TestIntegratorRWG
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorRWG.cc)
  target_link_libraries (TestIntegratorRWG ${TESTS_LIBS} )
  
  add_executable (TestIntegratorRWGAccuracy
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorRWGAccuracy.cc)
  target_link_libraries (TestIntegratorRWGAccuracy ${TESTS_LIBS} )

  add_executable (TestInverseDistance
    ${CMAKE_CURRENT_SOURCE_DIR}/TestInverseDistance.cc)
  target_link_libraries (TestInverseDistance ${TESTS_LIBS} )

  add_executable (TestSVDSolver
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSVDSolver.cc)
  target_link_libraries (TestSVDSolver ${TESTS_LIBS} )

  add_executable (TestBinaryTruncation
    ${CMAKE_CURRENT_SOURCE_DIR}/TestBinaryTruncation.cc)
  target_link_libraries (TestBinaryTruncation ${TESTS_LIBS} )

  add_executable (TestTriangles
    ${CMAKE_CURRENT_SOURCE_DIR}/TestTriangles.cc)
  target_link_libraries (TestTriangles ${TESTS_LIBS}  )

  add_executable (TestTypelists
    ${CMAKE_CURRENT_SOURCE_DIR}/TestTypelists.cc)
  target_link_libraries (TestTypelists ${TESTS_LIBS}  )

  add_executable (TestVisitor
    ${CMAKE_CURRENT_SOURCE_DIR}/TestVisitor.cc)
  target_link_libraries (TestVisitor ${TESTS_LIBS}  )

  add_executable (TestZonalHarmonics
    ${CMAKE_CURRENT_SOURCE_DIR}/TestZonalHarmonics.cc)
  target_link_libraries (TestZonalHarmonics ${TESTS_LIBS}  )

  kasper_install_executables (
    TestCubatureSpeedStandAlone
    TestDiskCapacitance
    TestElectromagnets
    TestElectromagnetViewer
    TestEMFileInterface
    TestGeometry
    TestAnisotropicPoisson
    TestIntegratorFunctionality
    TestIntegratorSpeed
    TestIntegratorSpeedWithKbdInput
    TestIntegratorRWG
    TestIntegratorRWGAccuracy
    TestInverseDistance
    TestSVDSolver
    TestBinaryTruncation
    TestTriangles
    TestTypelists
    TestVisitor
    TestZonalHarmonics
    )

############
### ROOT ###
############

  if (@PROJECT_NAME@_USE_ROOT)

  add_executable (TestDielectrics
    ${CMAKE_CURRENT_SOURCE_DIR}/TestDielectrics.cc)
  target_link_libraries (TestDielectrics ${TESTS_LIBS} )

  add_executable (TestIntegratorDistRatioLineSegmentROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorDistRatioLineSegmentROOT.cc)
  target_link_libraries (TestIntegratorDistRatioLineSegmentROOT ${TESTS_LIBS} )
  
  add_executable (TestIntegratorDistRatioRectangleROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorDistRatioRectangleROOT.cc)
  target_link_libraries (TestIntegratorDistRatioRectangleROOT ${TESTS_LIBS} )
  
  add_executable (TestIntegratorDistRatioTriangleROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorDistRatioTriangleROOT.cc)
  target_link_libraries (TestIntegratorDistRatioTriangleROOT ${TESTS_LIBS} )
  
  add_executable (TestVectorTypes
    ${CMAKE_CURRENT_SOURCE_DIR}/TestVectorTypes.cc)
  target_link_libraries (TestVectorTypes ${TESTS_LIBS} )

  kasper_install_executables (
    TestDielectrics
    TestIntegratorDistRatioLineSegmentROOT
    TestIntegratorDistRatioRectangleROOT
    TestIntegratorDistRatioTriangleROOT
    TestVectorTypes
  )

  endif (@PROJECT_NAME@_USE_ROOT)

################
### KMessage ###
################

  if (@PROJECT_NAME@_USE_KMESSAGE)

    add_executable (TestKMessageInterface
      ${CMAKE_CURRENT_SOURCE_DIR}/TestKMessageInterface.cc)
    target_link_libraries (TestKMessageInterface ${TESTS_LIBS}  )

  kasper_install_executables (
  TestKMessageInterface
    )

  endif (@PROJECT_NAME@_USE_KMESSAGE)

####################
### OpenCL + MPI ###
####################

#  if (@PROJECT_NAME@_USE_OPENCL AND @PROJECT_NAME@_USE_MPI)
#
#  add_executable (TestOpenCL
#    ${CMAKE_CURRENT_SOURCE_DIR}/TestOpenCL.cc)
#  target_link_libraries (TestOpenCL ${OPENCL_LIBRARIES} )

#  kasper_install_executables (
#  TestOpenCL
#    )

#  endif (@PROJECT_NAME@_USE_OPENCL AND @PROJECT_NAME@_USE_MPI)

##############
### OpenCL ###
##############

  if (@PROJECT_NAME@_USE_OPENCL)

  if (@PROJECT_NAME@_USE_ROOT)
    add_executable (TestIntegratorTriangleROOTOpenCL
      ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorTriangleROOTOpenCL.cc)
    target_link_libraries (TestIntegratorTriangleROOTOpenCL ${OPENCL_LIBRARIES} ${TESTS_LIBS} )
    
    add_executable (TestIntegratorRectangleROOTOpenCL
      ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorRectangleROOTOpenCL.cc)
    target_link_libraries (TestIntegratorRectangleROOTOpenCL ${OPENCL_LIBRARIES} ${TESTS_LIBS} )
    
    add_executable (TestIntegratorLineSegmentROOTOpenCL
      ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorLineSegmentROOTOpenCL.cc)
    target_link_libraries (TestIntegratorLineSegmentROOTOpenCL ${OPENCL_LIBRARIES} ${TESTS_LIBS} )
    
    kasper_install_executables (
      TestIntegratorTriangleROOTOpenCL
      TestIntegratorRectangleROOTOpenCL
      TestIntegratorLineSegmentROOTOpenCL
    )
    
  endif (@PROJECT_NAME@_USE_ROOT)

  add_executable (TestOpenCLPlugin
    ${CMAKE_CURRENT_SOURCE_DIR}/TestOpenCLPlugin.cc)
  target_link_libraries (TestOpenCLPlugin ${OPENCL_LIBRARIES} ${TESTS_LIBS} )
  
  add_executable (TestIntegratorFunctionalityOpenCL
    ${CMAKE_CURRENT_SOURCE_DIR}/TestIntegratorFunctionalityOpenCL.cc)
  target_link_libraries (TestIntegratorFunctionalityOpenCL ${OPENCL_LIBRARIES} ${TESTS_LIBS} )

  kasper_install_executables (
    TestIntegratorFunctionalityOpenCL
    TestOpenCLPlugin
  )

  endif (@PROJECT_NAME@_USE_OPENCL)

###############
### KGeoBag ###
###############

  if (@PROJECT_NAME@_USE_KGEOBAG)

    add_executable (TestCubeCapacitance
      ${CMAKE_CURRENT_SOURCE_DIR}/TestCubeCapacitance.cc)
    target_link_libraries (TestCubeCapacitance ${TESTS_LIBS})

    add_executable (TestSphereCapacitance
      ${CMAKE_CURRENT_SOURCE_DIR}/TestSphereCapacitance.cc)
    target_link_libraries (TestSphereCapacitance ${TESTS_LIBS})

  add_executable (TestSuperposition
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSuperposition.cc)
  target_link_libraries (TestSuperposition ${TESTS_LIBS} )

  add_executable (TestSymmetryGroups
    ${CMAKE_CURRENT_SOURCE_DIR}/TestSymmetryGroups.cc)
  target_link_libraries (TestSymmetryGroups ${TESTS_LIBS} )

#    add_executable (TestSphericalImageCharge
#      ${CMAKE_CURRENT_SOURCE_DIR}/TestSphericalImageCharge.cc)
#    target_link_libraries (TestSphericalImageCharge ${TESTS_LIBS} ${Kommon_LIBRARIES})
  
    kasper_install_executables (
      TestCubeCapacitance
      TestSphereCapacitance
      TestSuperposition
      TestSymmetryGroups
#      TestSphericalImageCharge
      )

  endif (@PROJECT_NAME@_USE_KGEOBAG)

#############
### PETSc ###
#############

  if (@PROJECT_NAME@_USE_PETSc)

set_property(
	SOURCE
	${CMAKE_CURRENT_SOURCE_DIR}/TestPETSc.cc
	APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_DATA_DIR="${DATA_INSTALL_DIR}/${@PROJECT_NAME@_DATA_DIR}"
)

    add_executable (TestPETSc
    ${CMAKE_CURRENT_SOURCE_DIR}/TestPETSc.cc)
    target_link_libraries (TestPETSc ${TESTS_LIBS} ${PETSC_LIBRARY})

    add_executable (TestPETSc1
    ${CMAKE_CURRENT_SOURCE_DIR}/TestPETSc1.cc)
    target_link_libraries (TestPETSc1 ${TESTS_LIBS} ${PETSC_LIBRARY})

    add_executable (TestPETSc2
    ${CMAKE_CURRENT_SOURCE_DIR}/TestPETSc2.cc)
    target_link_libraries (TestPETSc2 ${TESTS_LIBS} ${PETSC_LIBRARY})

    kasper_install_executables (
      TestPETSc
      TestPETSc1
      TestPETSc2
      )

  endif (@PROJECT_NAME@_USE_PETSc)

endif (@PROJECT_NAME@_ENABLE_TEST)
