set( APPLICATIONS_FIELDS_BASENAMES
    SimpleElectricFieldCalculator
    SimpleElectricFieldCalculatorAlongZaxis
    SimpleElectricFieldCalculatorFromFile
    SimpleElectricFieldCalculatorSpeedTest

    SimpleMagneticFieldCalculator
    SimpleMagneticFieldCalculatorAlongZaxis
    SimpleMagneticFieldCalculatorFromFile
    SimpleMagneticFieldCalculatorSpeedTest
)

set( APPLICATIONS_FIELDS_LIBRARIES
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
    KassiopeiaNavigators
    KassiopeiaTerminators
    KassiopeiaReaders
    KassiopeiaWriters
    KassiopeiaSimulation
    KassiopeiaBindings
)

if( KASPER_USE_BOOST )
    list( APPEND APPLICATIONS_FIELDS_LIBRARIES
        ${Boost_LIBRARIES}
    )
endif()

set( APPLICATIONS_FIELDS_PATH
	${CMAKE_CURRENT_SOURCE_DIR}/Source
)

kasper_internal_include_directories()
foreach( BASENAME ${APPLICATIONS_FIELDS_BASENAMES} )
	add_executable( ${BASENAME} ${APPLICATIONS_FIELDS_PATH}/${BASENAME}.cxx )
	target_link_libraries( ${BASENAME} ${APPLICATIONS_FIELDS_LIBRARIES} )
endforeach( BASENAME )

kasper_install_executables( ${APPLICATIONS_FIELDS_BASENAMES} )
