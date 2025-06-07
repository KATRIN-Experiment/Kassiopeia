# CMakeLists for KEMField/Interface/ChargeDensitySolvers
# Author: W. Gosda

# header files
set( CHARGEDENSITYSOLVER_ELECTRIC_HEADER_BASENAMES
    KChargeDensitySolver.hh
    KCachedChargeDensitySolver.hh
    KExplicitSuperpositionCachedChargeDensitySolver.hh
    KExplicitSuperpositionSolutionComponent.hh
    KGaussianEliminationChargeDensitySolver.hh
    KGaussSeidelChargeDensitySolver.hh
    KKrylovChargeDensitySolver.hh
    KKrylovChargeDensitySolverOld.hh
    KKrylovPreconditionerGenerator.hh
    KRobinHoodChargeDensitySolver.hh
)

set( CHARGEDENSITYSOLVER_MAGNETIC_HEADER_BASENAMES
    KMagneticChargeDensitySolver.hh
    KCachedMagneticChargeDensitySolver.hh
    KBiotSavartChargeDensitySolver.hh    
)

set( CHARGEDENSITYSOLVER_ELECTRIC_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Electric/include
)

set( CHARGEDENSITYSOLVER_MAGNETIC_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Magnetic/include
)

foreach( BASENAME ${CHARGEDENSITYSOLVER_ELECTRIC_HEADER_BASENAMES} )
    list( APPEND CHARGEDENSITYSOLVER_ELECTRIC_HEADER_FILENAMES ${CHARGEDENSITYSOLVER_ELECTRIC_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

foreach( BASENAME ${CHARGEDENSITYSOLVER_MAGNETIC_HEADER_BASENAMES} )
    list( APPEND CHARGEDENSITYSOLVER_MAGNETIC_HEADER_FILENAMES ${CHARGEDENSITYSOLVER_MAGNETIC_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )


# source files
set( CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_BASENAMES
    KChargeDensitySolver.cc
    KCachedChargeDensitySolver.cc
    KExplicitSuperpositionCachedChargeDensitySolver.cc
    KGaussianEliminationChargeDensitySolver.cc
    KGaussSeidelChargeDensitySolver.cc
    KKrylovChargeDensitySolver.cc
    KKrylovChargeDensitySolverOld.cc
    KKrylovPreconditionerGenerator.cc
    KRobinHoodChargeDensitySolver.cc
)

set( CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_BASENAMES
    KMagneticChargeDensitySolver.cc
    KCachedMagneticChargeDensitySolver.cc
    KBiotSavartChargeDensitySolver.cc    
)

set( CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Electric/src
)

set( CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/Magnetic/src
)

foreach( BASENAME ${CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_BASENAMES} )
    list( APPEND CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_FILENAMES ${CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

foreach( BASENAME ${CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_BASENAMES} )
    list( APPEND CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_FILENAMES ${CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )


# library
add_library( KEMFieldChargeDensitySolvers SHARED
    ${CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_FILENAMES} ${CHARGEDENSITYSOLVER_ELECTRIC_HEADER_FILENAMES}
    ${CHARGEDENSITYSOLVER_MAGNETIC_SOURCE_FILENAMES} ${CHARGEDENSITYSOLVER_MAGNETIC_HEADER_FILENAMES}
)
#set_target_properties( KEMFieldChargeDensitySolvers PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )

target_include_directories(KEMFieldChargeDensitySolvers
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Electric/include> $<INSTALL_INTERFACE:Electric/include>)
target_include_directories(KEMFieldChargeDensitySolvers
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Magnetic/include> $<INSTALL_INTERFACE:Magnetic/include>)


target_link_libraries( KEMFieldChargeDensitySolvers
    PUBLIC
        KEMFieldElectricBoundaryIntegrators
        KEMFileManipulation
        KEMLinearAlgebraSolvers
        KEMLinearAlgebraVisitors
        KFMUtility
)

if(KEMField_USE_ROOT)
    target_link_libraries(KEMFieldChargeDensitySolvers PUBLIC KEMRootPlugin)
endif(KEMField_USE_ROOT)

if(KEMField_USE_OPENCL)
    target_link_libraries(KEMFieldChargeDensitySolvers PUBLIC KEMOpenCLPlugin)
endif(KEMField_USE_OPENCL)

if(KEMField_USE_VTK)
    target_link_libraries(KEMFieldChargeDensitySolvers PUBLIC KEMVTKPlugin)
endif(KEMField_USE_VTK)

# install
kasper_install_headers( ${CHARGEDENSITYSOLVER_ELECTRIC_HEADER_FILENAMES} )
kasper_install_headers( ${CHARGEDENSITYSOLVER_MAGNETIC_HEADER_FILENAMES} )
kasper_install_libraries( KEMFieldChargeDensitySolvers )

# test executables
#add_executable( KChargeDensitySolver_test Electric/test/KChargeDensitySolver_test.cc )
#target_link_libraries( KChargeDensitySolver_test
#  ${KGeoBag_LIBRARIES}
#  KEMFieldChargeDensitySolvers
#)
#add_executable( KMagneticChargeDensitySolver_test Magnetic/test/KMagneticChargeDensitySolver_test.cc )
#target_link_libraries( KMagneticChargeDensitySolver_test
#  ${KGeoBag_LIBRARIES}
#  KEMFieldChargeDensitySolvers
#)


#kasper_install_executables( KChargeDensitySolver_test )
#kasper_install_executables( KMagneticChargeDensitySolver_test )
