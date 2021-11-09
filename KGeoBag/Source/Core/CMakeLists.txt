# headers
set( CORE_HEADER_FILES
    Include/KGCoreMessage.hh
    Include/KGCore.hh
    Include/KGVisitor.hh
    Include/KGBoundary.hh
    Include/KGArea.hh
    Include/KGVolume.hh
    Include/KGSurface.hh
    Include/KGSurfaceFunctions.hh
    Include/KGExtensibleSurface.hh
    Include/KGExtendedSurface.hh
    Include/KGExtendedSurfaceFunctions.hh
    Include/KGSpace.hh
    Include/KGSpaceFunctions.hh
    Include/KGExtensibleSpace.hh
    Include/KGExtendedSpace.hh
    Include/KGExtendedSpaceFunctions.hh
    Include/KGInterface.hh
    Include/KGTypelist.hh
)

# source
set( CORE_SOURCE_FILES
    Source/KGCoreMessage.cc
    Source/KGVisitor.cc
    Source/KGBoundary.cc
    Source/KGArea.cc
    Source/KGVolume.cc
    Source/KGSurface.cc
    Source/KGExtensibleSurface.cc
    Source/KGSpace.cc
    Source/KGExtensibleSpace.cc
    Source/KGInterface.cc
)

# target
add_library( KGeoBagCore SHARED
    ${CORE_SOURCE_FILES} ${CORE_HEADER_FILES}
)
target_include_directories( KGeoBagCore
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Include> $<INSTALL_INTERFACE:include>
)
target_link_libraries( KGeoBagCore
    PUBLIC
        Kommon
        KGeoBagMath
)

if( KGeoBag_ENABLE_DEBUG )
    target_compile_definitions( KGeoBagCore PUBLIC KGeoBag_ENABLE_DEBUG )
endif( KGeoBag_ENABLE_DEBUG )

if( KGeoBag_USE_GSL )
    target_compile_definitions( KGeoBagCore PUBLIC KGeoBag_USE_GSL )
endif( KGeoBag_USE_GSL )

if( KGeoBag_USE_BOOST )
    target_compile_definitions( KGeoBagCore PUBLIC KGeoBag_USE_BOOST )
endif()

# install
kasper_install_headers( ${CORE_HEADER_FILES} )
kasper_install_libraries( KGeoBagCore )
