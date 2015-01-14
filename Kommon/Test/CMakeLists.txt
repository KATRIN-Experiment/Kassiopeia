kasper_internal_include_directories(
	${CMAKE_CURRENT_SOURCE_DIR}
)

set( TEST_CONFIG_FILES
    TestXMLTokenizer.xml
    TestXMLVariables.xml
    TestXMLFormulas.xml
    TestXMLIncludes.xml
    TestXMLLoops.xml
    TestXMLConditions.xml
    TestXMLElements.xml
    TestXMLSerialization.xml
    TestXMLPrint.xml
)
kasper_install_config( ${TEST_CONFIG_FILES} )

if( Kommon_USE_ROOT )

	set( TEST_APPLICATION_NAMES
		TestXMLTokenizer
		TestXMLVariables
		TestXMLFormulas
		TestXMLIncludes
		TestXMLLoops
		TestXMLConditions
		TestXMLElements
		TestXMLSerialization
		TestXMLPrint
	)
	foreach( NAME ${TEST_APPLICATION_NAMES} )
		add_executable( ${NAME} ${NAME}.cxx )
		target_link_libraries( ${NAME} KommonCore KommonRoot )
		kasper_install_executables( ${NAME} )
	endforeach()

endif( Kommon_USE_ROOT )
