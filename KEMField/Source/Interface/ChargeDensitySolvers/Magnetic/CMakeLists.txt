# CMakeLists for KEMField/Interface/ChargeDensitySolvers/Magnetic
# Author: P. Slocum

# header files
set( CHARGEDENSITYSOLVER_MAGNETIC_HEADER_BASENAMES
    KMagneticChargeDensitySolver.hh
)

set( CHARGEDENSITYSOLVER_MAGNETIC_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

foreach( BASENAME ${CHARGEDENSITYSOLVER_MAGNETIC_HEADER_BASENAMES} )
    list( APPEND CHARGEDENSITYSOLVER_MAGNETIC_HEADER_FILENAMES ${CHARGEDENSITYSOLVER_MAGNETIC_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_BASENAMES
    KMagneticChargeDensitySolver.cc
)

set( CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
foreach( BASENAME ${CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_BASENAMES} )
    list( APPEND CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_FILENAMES ${CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KEMFieldMagneticChargeDensitySolvers SHARED
    ${CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_FILENAMES} ${CHARGEDENSITYSOLVER_MAGNETIC_HEADER_FILENAMES})
#set_target_properties( KEMFieldMagneticChargeDensitySolvers PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )

target_include_directories(KEMFieldMagneticChargeDensitySolvers
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KEMFieldMagneticChargeDensitySolvers
    PUBLIC
        KEMFileManipulation
        KEMLinearAlgebraSolvers
        KEMLinearAlgebraVisitors
        KFMUtility       
)

if(KEMField_USE_ROOT)
    target_link_libraries(KEMFieldMagneticChargeDensitySolvers PUBLIC KEMRootPlugin)
endif(KEMField_USE_ROOT)

if(KEMField_USE_OPENCL)
    target_link_libraries(KEMFieldMagneticChargeDensitySolvers PUBLIC KEMOpenCLPlugin)
endif(KEMField_USE_OPENCL)

if(KEMField_USE_VTK)
    target_link_libraries(KEMFieldMagneticChargeDensitySolvers PUBLIC KEMVTKPlugin)
endif(KEMField_USE_VTK)

# install
kasper_install_headers( ${CHARGEDENSITYSOLVER_MAGNETIC_HEADER_FILENAMES} )
kasper_install_libraries( KEMFieldMagneticChargeDensitySolvers )
