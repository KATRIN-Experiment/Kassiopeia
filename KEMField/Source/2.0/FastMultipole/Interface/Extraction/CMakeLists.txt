# CMakeLists for KEMField/FastMultipole/Subdivision
# Author: J. P. Barrett

kasper_internal_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set (INTERFACE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMSurfaceToPointCloudConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticBasisDataExtractor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticSurfaceConverter.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElementAspectRatioExtractor.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMPointCloudContainer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMBoundingBallContainer.hh
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticBasisDataContainer.hh
#    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMElectrostaticElementContainerFlyweight.hh
)

set (INTERFACE_SOURCEFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMSurfaceToPointCloudConverter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticSurfaceConverter.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElementAspectRatioExtractor.cc
#    ${CMAKE_CURRENT_SOURCE_DIR}/src/KFMElectrostaticElementContainerFlyweight.cc
)

##################################################

add_library (KFMInterfaceExtraction SHARED ${INTERFACE_SOURCEFILES})
target_link_libraries (KFMInterfaceExtraction KEMMath KEMSurfaces KFMCore KFMElectrostatics KEMIntegratingFieldSolver)

kasper_install_headers (${INTERFACE_HEADERFILES})
kasper_install_libraries (KFMInterfaceExtraction)
