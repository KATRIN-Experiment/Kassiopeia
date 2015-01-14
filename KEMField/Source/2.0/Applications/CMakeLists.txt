# CMakeLists for KEMField/Applications
# Author: T.J. Corona

kasper_include_default_dirs()

set (APPLICATIONS_LIBS
  ${ROOT_LIBRARIES}
  ${GSL_LIBRARIES}
  KEMCore
  KEMHashGenerator
  KEMStructuredASCII
  KEMMath
  KEMSurfaces
  KEMIO
  KEMFileManipulation
  KEMElectrostaticBoundaryIntegrals
  KEMElectromagnets
  KEMIntegratingFieldSolver
  KEMZHGenerator
  KEMZHSolver
  KEMVisualization
  KFMMathUtilities
  )

if (@PROJECT_NAME@_USE_VTK)
  list (APPEND APPLICATIONS_LIBS
    KEMVTKPlugin
    )
endif (@PROJECT_NAME@_USE_VTK)

if (@PROJECT_NAME@_USE_ROOT)
  list (APPEND APPLICATIONS_LIBS
    KEMRootPlugin
    )
endif (@PROJECT_NAME@_USE_ROOT)

if (@PROJECT_NAME@_USE_OPENCL)
  list (APPEND APPLICATIONS_LIBS
    KEMOpenCLPlugin
    )
endif (@PROJECT_NAME@_USE_OPENCL)

if (@PROJECT_NAME@_USE_KGEOBAG)
    kasper_find_module(KGeoBag)
  list (APPEND APPLICATIONS_LIBS
    KEMGeoBagPlugin
    )
endif (@PROJECT_NAME@_USE_KGEOBAG)

if (@PROJECT_NAME@_USE_PETSc)
  list (APPEND APPLICATIONS_LIBS
    KEMPETScPlugin
    )
endif (@PROJECT_NAME@_USE_PETSc)

if (@PROJECT_NAME@_USE_VTK)
  find_package(VTK REQUIRED)
  include(${VTK_USE_FILE})
  list (APPEND APPLICATIONS_LIBS
    ${VTK_LIBRARIES}
    )
endif (@PROJECT_NAME@_USE_VTK)

set_property(
  SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitor.cc
  APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_DATA_DIR="${@PROJECT_NAME@_CONFIG_INSTALL_DIR}"
  )

set_property(
  SOURCE
  ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitor.cc
  APPEND PROPERTY COMPILE_DEFINITIONS DEFAULT_OUTPUT_DIR="${@PROJECT_NAME@_OUTPUT_INSTALL_DIR}"
  )

add_executable (ComputeChargeDensities
  ${CMAKE_CURRENT_SOURCE_DIR}/ComputeChargeDensities.cc)
target_link_libraries (ComputeChargeDensities ${APPLICATIONS_LIBS} )

add_executable (ComputeSphericalCapacitor
  ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitor.cc)
target_link_libraries (ComputeSphericalCapacitor ${APPLICATIONS_LIBS} )

add_executable (TransferEMElement
  ${CMAKE_CURRENT_SOURCE_DIR}/TransferEMElement.cc)
target_link_libraries (TransferEMElement ${APPLICATIONS_LIBS} )

add_executable (HashEMGeometry
  ${CMAKE_CURRENT_SOURCE_DIR}/HashEMGeometry.cc)
target_link_libraries (HashEMGeometry ${APPLICATIONS_LIBS} )

add_executable (InspectEMFile
  ${CMAKE_CURRENT_SOURCE_DIR}/InspectEMFile.cc)
target_link_libraries (InspectEMFile ${APPLICATIONS_LIBS} )

kasper_install_executables (
  ComputeChargeDensities
  ComputeSphericalCapacitor
  TransferEMElement
  HashEMGeometry
  InspectEMFile
  )

if (@PROJECT_NAME@_USE_VTK)

  add_executable (VisualizeEMGeometry
    ${CMAKE_CURRENT_SOURCE_DIR}/VisualizeEMGeometry.cc)
  target_link_libraries (VisualizeEMGeometry ${APPLICATIONS_LIBS} )

  kasper_install_executables (
    VisualizeEMGeometry
    )

endif (@PROJECT_NAME@_USE_VTK)
