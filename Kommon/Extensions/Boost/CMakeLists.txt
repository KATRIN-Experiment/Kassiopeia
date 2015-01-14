# headers
set( KOMMON_BOOST_HEADER_FILENAMES
	Utility/KAccumulators.h
	Utility/KValueCache.h
	Utility/KFunctionCache.h
	Utility/KStringUtils.h
	Utility/KNumeric.h
	Utility/KFastMath.h
	Utility/KForeach.h
	Utility/KMathIntegrator.h
	Utility/KMathIntegrator2D.h
	Utility/KMathShepardInterpolator.h
)

# sources
set( KOMMON_BOOST_SOURCE_FILENAMES
	Utility/KFunctionCache.cxx
)

# external
find_package( Boost 1.42.0 REQUIRED COMPONENTS system date_time regex program_options thread )
kasper_external_include_directories(
	${Boost_INCLUDE_DIR}
)
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Utility
	${CMAKE_CURRENT_SOURCE_DIR}/Vault
)

# link
add_library( KommonBoost SHARED ${KOMMON_BOOST_SOURCE_FILENAMES} )
target_link_libraries( KommonBoost KommonCore ${Boost_LIBRARIES} )

# install
kasper_install_libraries( KommonBoost )
kasper_install_headers( ${KOMMON_BOOST_HEADER_FILENAMES} )

# bundled boost headers
install( DIRECTORY Vault/boost DESTINATION "${INCLUDE_INSTALL_DIR}" )