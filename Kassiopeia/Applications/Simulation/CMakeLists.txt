set( MAIN_LIBRARIES
    ${Kommon_LIBRARIES}
    ${KGeoBag_LIBRARIES}
    ${KEMField_LIBRARIES}
    KassiopeiaUtility
    KassiopeiaMath
    KassiopeiaObjects
    KassiopeiaOperators
    KassiopeiaFields
    KassiopeiaTrajectories
    KassiopeiaInteractions
    KassiopeiaTerminators
    KassiopeiaWriters
    KassiopeiaVisualization
    KassiopeiaReaders
    KassiopeiaSimulation
    KassiopeiaBindings
)

kasper_internal_include_directories()

# target
if(APPLE)
	add_executable( Kassiopeia MACOSX_BUNDLE ${CMAKE_CURRENT_SOURCE_DIR}/Source/Kassiopeia.cxx )
else(APPLE)
	add_executable( Kassiopeia ${CMAKE_CURRENT_SOURCE_DIR}/Source/Kassiopeia.cxx )
endif(APPLE)
target_link_libraries (Kassiopeia ${MAIN_LIBRARIES})
kasper_install_executables( Kassiopeia )
