# CMakeLists for KEMField/Interface/ChargeDensitySolvers/Electric
# Author: W. Gosda

# header files
set( CHARGEDENSITYSOLVER_ELECTRIC_HEADER_BASENAMES
    KChargeDensitySolver.hh
    KCachedChargeDensitySolver.hh
    KExplicitSuperpositionCachedChargeDensitySolver.hh
    KExplicitSuperpositionSolutionComponent.hh
    KGaussianEliminationChargeDensitySolver.hh
    KKrylovChargeDensitySolver.hh
    KKrylovChargeDensitySolverOld.hh
    KKrylovPreconditionerGenerator.hh
    KRobinHoodChargeDensitySolver.hh
)

set( CHARGEDENSITYSOLVER_ELECTRIC_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
foreach( BASENAME ${CHARGEDENSITYSOLVER_ELECTRIC_HEADER_BASENAMES} )
    list( APPEND CHARGEDENSITYSOLVER_ELECTRIC_HEADER_FILENAMES ${CHARGEDENSITYSOLVER_ELECTRIC_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_BASENAMES
    KChargeDensitySolver.cc
    KCachedChargeDensitySolver.cc
    KExplicitSuperpositionCachedChargeDensitySolver.cc
    KGaussianEliminationChargeDensitySolver.cc
    KKrylovChargeDensitySolver.cc
    KKrylovChargeDensitySolverOld.cc
    KKrylovPreconditionerGenerator.cc
    KRobinHoodChargeDensitySolver.cc
)

set( CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
foreach( BASENAME ${CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_BASENAMES} )
    list( APPEND CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_FILENAMES ${CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
kasper_internal_include_directories( ${CHARGEDENSITYSOLVER_ELECTRIC_HEADER_PATH} )
add_library( KEMFieldChargeDensitySolvers SHARED ${CHARGEDENSITYSOLVER_ELECTRIC_SOURCE_FILENAMES} ${CHARGEDENSITYSOLVER_ELECTRIC_HEADER_FILENAMES} )
set_target_properties( KEMFieldChargeDensitySolvers PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KEMFieldChargeDensitySolvers
    ${KGeoBag_LIBRARIES}
    KEMElectrostaticBoundaryIntegrals
    KEMFieldElectricBoundaryIntegrators
    KEMFileManipulation
    KEMFieldExceptions
    KEMHashGenerator
    KEMLinearAlgebraSolvers
    KFMUtility
)

# install
kasper_install_headers( ${CHARGEDENSITYSOLVER_ELECTRIC_HEADER_FILENAMES} )
kasper_install_libraries( KEMFieldChargeDensitySolvers )

# test executable
#add_executable( KChargeDensitySolver_test test/KChargeDensitySolver_test.cc )
#target_link_libraries( KChargeDensitySolver_test
#  ${KGeoBag_LIBRARIES}
#  KEMFieldChargeDensitySolvers
#)

#kasper_install_executables( KChargeDensitySolver_test )
