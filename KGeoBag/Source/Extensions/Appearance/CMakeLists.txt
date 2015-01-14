# headers
set( APPEARANCE_HEADER_FILES
    Include/KGRGBColor.hh
    Include/KGRGBAColor.hh
	Include/KGAppearance.hh
)

# sources
set( APPEARANCE_SOURCE_FILES
    Source/KGRGBColor.cc
    Source/KGRGBAColor.cc
	Source/KGAppearance.cc
)

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
add_library( KGeoBagAppearance SHARED ${APPEARANCE_SOURCE_FILES} ${APPEARANCE_HEADER_FILES} )
target_link_libraries( KGeoBagAppearance
	KGeoBagCore
	${Kommon_LIBRARIES}
)

# install
kasper_install_headers( ${APPEARANCE_HEADER_FILES} )
kasper_install_libraries( KGeoBagAppearance )


