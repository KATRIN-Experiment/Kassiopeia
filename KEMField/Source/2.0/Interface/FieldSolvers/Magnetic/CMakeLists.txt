# CMakeLists for KEMField/Interface/FieldSolvers/Magnetic
# Author: W. Gosda

# header files
set( FIELDSOLVER_MAGNETIC_HEADER_BASENAMES
    KMagneticFieldSolver.hh
    KIntegratingMagnetostaticFieldSolver.hh
    KZonalHarmonicMagnetostaticFieldSolver.hh
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
)

set( FIELDSOLVER_MAGNETIC_SOURCE_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/src
)

foreach( BASENAME ${FIELDSOLVER_MAGNETIC_SOURCE_BASENAMES} )
	list( APPEND FIELDSOLVER_MAGNETIC_SOURCE_FILENAMES ${FIELDSOLVER_MAGNETIC_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# link libraries
set (FIELDSOLVER_MAGNETIC_LINK_LIBRARIES
    KEMElectromagnets
    KEMZHSolver
)

# library
kasper_internal_include_directories( ${FIELDSOLVER_MAGNETIC_HEADER_PATH} )
add_library( KEMMagneticFieldSolvers SHARED ${FIELDSOLVER_MAGNETIC_SOURCE_FILENAMES} )
set_target_properties( KEMMagneticFieldSolvers PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KEMMagneticFieldSolvers 
    ${FIELDSOLVER_MAGNETIC_LINK_LIBRARIES}
)

# install
kasper_install_headers( ${FIELDSOLVER_MAGNETIC_HEADER_FILENAMES} )
kasper_install_libraries( KEMMagneticFieldSolvers )
