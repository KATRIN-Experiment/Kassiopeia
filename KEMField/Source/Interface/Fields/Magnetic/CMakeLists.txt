# CMakeLists for KEMField/Interface/Fields/Magnetic
# Author: W. Gosda

# header files
set (FIELDS_MAGNETIC_HEADER_BASENAMES
    KMagneticField.hh
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
kasper_internal_include_directories( ${FIELDS_MAGNETIC_HEADER_PATH} )
add_library( KEMFieldsMagnetic SHARED ${FIELDS_MAGNETIC_SOURCE_FILENAMES} ${FIELDS_MAGNETIC_HEADER_FILENAMES} )
set_target_properties( KEMFieldsMagnetic PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KEMFieldsMagnetic
    KEMFieldChargeDensitySolvers
    KEMMath KEMCore ${Kommon_LIBRARIES} ${GSL_LIBRARIES}
)

# install
kasper_install_headers( ${FIELDS_MAGNETIC_HEADER_FILENAMES} )
kasper_install_libraries( KEMFieldsMagnetic )
