# header files
set( FIELDS_HEADER_BASENAMES
	KSFieldsMessage.h
	KSFieldElectromagnet.h
	KSFieldElectrostatic.h
	KSFieldKrylovBEMSolver.h
	KSFieldMagneticConstant.h
	KSFieldMagneticDipole.h
	KSFieldElectricConstant.h
	KSFieldElectricQuadrupole.h
	KSFieldMagneticSuperPosition.h
	KSFieldMagneticRamped.h
	KSFieldElectricRamped.h
)
if( Kassiopeia_USE_VTK )
	set( FIELDS_HEADER_BASENAMES
		${FIELDS_HEADER_BASENAMES}
		KSFieldElectricPotentialmap.h
	)
endif( Kassiopeia_USE_VTK )
set( FIELDS_HEADER_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)
foreach( BASENAME ${FIELDS_HEADER_BASENAMES} )
	list( APPEND FIELDS_HEADER_FILENAMES ${FIELDS_HEADER_PATH}/${BASENAME} )
endforeach( BASENAME )

# source files
set( FIELDS_SOURCE_BASENAMES
	KSFieldsMessage.cxx
	KSFieldElectromagnet.cxx
	KSFieldElectrostatic.cxx
	KSFieldKrylovBEMSolver.cxx
	KSFieldMagneticConstant.cxx
	KSFieldMagneticDipole.cxx
	KSFieldElectricConstant.cxx
	KSFieldElectricQuadrupole.cxx
	KSFieldMagneticSuperPosition.cxx
	KSFieldMagneticRamped.cxx
	KSFieldElectricRamped.cxx
)
if( Kassiopeia_USE_VTK )
	set( FIELDS_SOURCE_BASENAMES
		${FIELDS_SOURCE_BASENAMES}
		KSFieldElectricPotentialmap.cxx
	)
endif( Kassiopeia_USE_VTK )
set( FIELDS_SOURCE_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)
foreach( BASENAME ${FIELDS_SOURCE_BASENAMES} )
	list( APPEND FIELDS_SOURCE_FILENAMES ${FIELDS_SOURCE_PATH}/${BASENAME} )
endforeach( BASENAME )

# library
kasper_internal_include_directories( ${FIELDS_HEADER_PATH} )
add_library( KassiopeiaFields SHARED ${FIELDS_SOURCE_FILENAMES} )
set_target_properties( KassiopeiaFields PROPERTIES COMPILE_FLAGS -Wno-overloaded-virtual )
target_link_libraries( KassiopeiaFields
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    ${KEMField_LIBRARIES}
    KassiopeiaUtility
    KassiopeiaObjects
    KassiopeiaOperators
)

# install
kasper_install_headers( ${FIELDS_HEADER_FILENAMES} )
kasper_install_libraries( KassiopeiaFields )
