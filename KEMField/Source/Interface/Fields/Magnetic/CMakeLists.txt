# CMakeLists for KEMField/Interface/Fields/Magnetic
# Author: W. Gosda

# header files
set (FIELDS_MAGNETIC_HEADER_BASENAMES
    KMagneticField.hh
    KMagnetostaticImportField.hh
    KMagnetostaticExpoField.hh
    KStaticElectromagnetField.hh
    KMagnetostaticField.hh
    KMagnetostaticConstantField.hh
    KMagneticDipoleField.hh
    KMagneticSuperpositionField.hh
    KRampedMagneticField.hh
)

set( FIELDS_MAGNETIC_HEADER_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

foreach( BASENAME ${FIELDS_MAGNETIC_HEADER_BASENAMES} )
    list( APPEND FIELDS_MAGNETIC_HEADER_FILENAMES ${FIELDS_MAGNETIC_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( FIELDS_MAGNETIC_SOURCE_BASENAMES
    KStaticElectromagnetField.cc
    KMagneticDipoleField.cc
    KMagnetostaticImportField.cc
    KMagnetostaticExpoField.cc
    KMagnetostaticConstantField.cc
    KMagneticSuperpositionField.cc
    KRampedMagneticField.cc
)

set( FIELDS_MAGNETIC_SOURCE_PATH
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
foreach( BASENAME ${FIELDS_MAGNETIC_SOURCE_BASENAMES} )
    list( APPEND FIELDS_MAGNETIC_SOURCE_FILENAMES ${FIELDS_MAGNETIC_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
add_library( KEMFieldsMagnetic SHARED
    ${FIELDS_MAGNETIC_SOURCE_FILENAMES} ${FIELDS_MAGNETIC_HEADER_FILENAMES} )
#set_target_properties( KEMFieldsMagnetic PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )

target_include_directories(KEMFieldsMagnetic
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>)
target_link_libraries( KEMFieldsMagnetic
    PUBLIC
        KommonCore
        KEMCore
        KEMMath
        KEMFieldExceptions
        KEMFileManipulation
        KEMMagneticFieldSolvers
        KEMZHSolver
)

# install
kasper_install_headers( ${FIELDS_MAGNETIC_HEADER_FILENAMES} )
kasper_install_libraries( KEMFieldsMagnetic )
