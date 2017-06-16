set( MAIN_LIBRARIES
    ${Kommon_LIBRARIES}
    ${Kommon_Vtk_LIBRARIES}
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
add_executable( Kassiopeia MACOSX_BUNDLE ${CMAKE_CURRENT_SOURCE_DIR}/Source/Kassiopeia.cxx )
target_link_libraries (Kassiopeia ${MAIN_LIBRARIES})

kasper_install_executables( Kassiopeia )
