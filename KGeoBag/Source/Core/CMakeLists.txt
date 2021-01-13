# headers
set( CORE_HEADER_FILES
    Include/KGCoreMessage.hh
    Include/KGCore.hh
    Include/KGVisitor.hh
    Include/KGBoundary.hh
    Include/KGArea.hh
    Include/KGVolume.hh
    Include/KGSurface.hh
    Include/KGExtensibleSurface.hh
    Include/KGExtendedSurface.hh
    Include/KGSpace.hh
    Include/KGExtensibleSpace.hh
    Include/KGExtendedSpace.hh
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

# internal
kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/Include
)

# target
add_library( KGeoBagCore SHARED
	${CORE_SOURCE_FILES} ${CORE_HEADER_FILES}
)
target_link_libraries( KGeoBagCore
    KGeoBagMath
)

# install
kasper_install_headers( ${CORE_HEADER_FILES} )
kasper_install_libraries( KGeoBagCore )
