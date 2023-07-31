# CMakeLists for KEMField/Interface/FieldSolvers/Magnetic
# Author: W. Gosda

# header files
set( FIELDSOLVER_MAGNETIC_HEADER_BASENAMES
    KMagneticFieldSolver.hh
    KIntegratingMagnetostaticFieldSolver.hh
    KZonalHarmonicMagnetostaticFieldSolver.hh
    KMagfieldCoilsFieldSolver.hh
)

set( FIELDSOLVER_MAGNETIC_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

foreach( BASENAME ${FIELDSOLVER_MAGNETIC_HEADER_BASENAMES} )
    list( APPEND FIELDSOLVER_MAGNETIC_HEADER_FILENAMES ${FIELDSOLVER_MAGNETIC_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( FIELDSOLVER_MAGNETIC_SOURCE_BASENAMES
    KIntegratingMagnetostaticFieldSolver.cc
    KZonalHarmonicMagnetostaticFieldSolver.cc
    KMagfieldCoilsFieldSolver.cc
)

set( FIELDSOLVER_MAGNETIC_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

foreach( BASENAME ${FIELDSOLVER_MAGNETIC_SOURCE_BASENAMES} )
    list( APPEND FIELDSOLVER_MAGNETIC_SOURCE_FILENAMES ${FIELDSOLVER_MAGNETIC_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KEMMagneticFieldSolvers SHARED
    ${FIELDSOLVER_MAGNETIC_SOURCE_FILENAMES} ${FIELDSOLVER_MAGNETIC_HEADER_FILENAMES} )
target_include_directories(KEMMagneticFieldSolvers
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KEMMagneticFieldSolvers
    PRIVATE
        KEMFileManipulation
    PUBLIC
        KEMElectromagnets
        KEMMagfieldCoils
        KEMZHSolver
)
#set_target_properties( KEMMagneticFieldSolvers PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )

# install
kasper_install_headers( ${FIELDSOLVER_MAGNETIC_HEADER_FILENAMES} )
kasper_install_libraries( KEMMagneticFieldSolvers )
