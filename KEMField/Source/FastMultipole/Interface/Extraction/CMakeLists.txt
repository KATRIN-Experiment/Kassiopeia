# CMakeLists for KEMField/FastMultipole/Subdivision
# Author: J. P. Barrett

set (INTERFACE_HEADERFILES
    ${CMAKE_CURRENT_SOURCE_DIR}/include/KFMTreeSurfaceOrderingPredicate.hh
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

add_library (KFMInterfaceExtraction SHARED
    ${INTERFACE_SOURCEFILES} ${INTERFACE_HEADERFILES})
target_include_directories(KFMInterfaceExtraction
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries (KFMInterfaceExtraction
    PUBLIC
        KEMSurfaces
        KFMCore
        KFMElectrostatics
)

kasper_install_headers (${INTERFACE_HEADERFILES})
kasper_install_libraries (KFMInterfaceExtraction)
