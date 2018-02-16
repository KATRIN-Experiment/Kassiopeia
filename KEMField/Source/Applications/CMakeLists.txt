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
  KEMZHGenerator
  KEMZHSolver
  KEMVisualization
  KFMMathUtilities
  )

if (@PROJECT_NAME@_USE_ROOT)
  kasper_external_include_directories( ${ROOT_INCLUDE_DIRS} )
  list (APPEND APPLICATIONS_LIBS
    KEMRootPlugin
    )
endif (@PROJECT_NAME@_USE_ROOT)

if (@PROJECT_NAME@_USE_OPENCL)
  kasper_external_include_directories( ${OPENCL_INCLUDE_DIRS} )
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
  list (APPEND APPLICATIONS_LIBS KEMVTKPlugin ${Kommon_Vtk_LIBRARIES} )
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

add_executable (ComputeChargeDensitiesFromElcd33File
  ${CMAKE_CURRENT_SOURCE_DIR}/ComputeChargeDensitiesFromElcd33File.cc)
target_link_libraries (ComputeChargeDensitiesFromElcd33File ${APPLICATIONS_LIBS} )

add_executable (ComputeSphericalCapacitor
  ${CMAKE_CURRENT_SOURCE_DIR}/ComputeSphericalCapacitor.cc)
target_link_libraries (ComputeSphericalCapacitor ${APPLICATIONS_LIBS} )

add_executable (KdbConverter
  ${CMAKE_CURRENT_SOURCE_DIR}/KdbConverter.cc)
target_link_libraries (KdbConverter ${APPLICATIONS_LIBS} )

add_executable (TransferEMElement
  ${CMAKE_CURRENT_SOURCE_DIR}/TransferEMElement.cc)
target_link_libraries (TransferEMElement ${APPLICATIONS_LIBS} )

add_executable (HashEMGeometry
  ${CMAKE_CURRENT_SOURCE_DIR}/HashEMGeometry.cc)
target_link_libraries (HashEMGeometry ${APPLICATIONS_LIBS} )

add_executable (InspectEMFile
  ${CMAKE_CURRENT_SOURCE_DIR}/InspectEMFile.cc)
target_link_libraries (InspectEMFile ${APPLICATIONS_LIBS} )

add_executable (WriteKbdToAscii
  ${CMAKE_CURRENT_SOURCE_DIR}/WriteKbdToAscii.cc)
target_link_libraries (WriteKbdToAscii ${APPLICATIONS_LIBS} )

add_executable (WriteAsciiToPCD
  ${CMAKE_CURRENT_SOURCE_DIR}/WriteAsciiToPCD.cc)
target_link_libraries (WriteAsciiToPCD ${APPLICATIONS_LIBS} )

kasper_install_executables (
  ComputeChargeDensities
  ComputeChargeDensitiesFromElcd33File
  ComputeSphericalCapacitor
  KdbConverter
  TransferEMElement
  HashEMGeometry
  InspectEMFile
  WriteKbdToAscii
  WriteAsciiToPCD
  )

if (@PROJECT_NAME@_USE_ROOT)

  add_executable (AspectRatioFromKbdROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/AspectRatioFromKbdROOT.cc)
  target_link_libraries (AspectRatioFromKbdROOT ${APPLICATIONS_LIBS} )

    add_executable (CompareFieldsAndPotentialsROOT
      ${CMAKE_CURRENT_SOURCE_DIR}/CompareFieldsAndPotentialsROOT.cc)
    target_link_libraries (CompareFieldsAndPotentialsROOT ${APPLICATIONS_LIBS} )

  add_executable (DistanceRatioFromKbdROOT
    ${CMAKE_CURRENT_SOURCE_DIR}/DistanceRatioFromKbdROOT.cc)
  target_link_libraries (DistanceRatioFromKbdROOT ${APPLICATIONS_LIBS} )

  kasper_install_executables (
    AspectRatioFromKbdROOT
    CompareFieldsAndPotentialsROOT
    DistanceRatioFromKbdROOT
  )

endif (@PROJECT_NAME@_USE_ROOT)




if (@PROJECT_NAME@_USE_VTK)

  add_executable (VisualizeEMGeometry
    ${CMAKE_CURRENT_SOURCE_DIR}/VisualizeEMGeometry.cc)
  target_link_libraries (VisualizeEMGeometry ${APPLICATIONS_LIBS} )

  kasper_install_executables (
    VisualizeEMGeometry
    )

install (DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ApplicationFiles
  DESTINATION ${@PROJECT_NAME@_CONFIG_INSTALL_DIR}
  PATTERN "*.gitignore" EXCLUDE
  )

endif (@PROJECT_NAME@_USE_VTK)

#PCL Stuff
#cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

#project(greedy_projection)

#find_package(PCL 1.2 REQUIRED)

#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

#add_executable (greedy_projection ${CMAKE_CURRENT_SOURCE_DIR}/greedy_projection.cpp)
#target_link_libraries (greedy_projection ${PCL_LIBRARIES})

#kasper_install_executables (
#  greedy_projection
#  )
