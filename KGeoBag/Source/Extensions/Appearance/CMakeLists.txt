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

# target
add_library( KGeoBagAppearance SHARED
    ${APPEARANCE_SOURCE_FILES} ${APPEARANCE_HEADER_FILES}
)
target_include_directories( KGeoBagAppearance
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include> $<INSTALL_INTERFACE:include>
)
target_link_libraries( KGeoBagAppearance
    PUBLIC
       KGeoBagCore
)

# install
kasper_install_headers( ${APPEARANCE_HEADER_FILES} )
kasper_install_libraries( KGeoBagAppearance )
