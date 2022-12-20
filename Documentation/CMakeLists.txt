if(BUILD_DOC)
    add_subdirectory(DeveloperGuide)
    add_subdirectory(ComponentDocumentation)
endif(BUILD_DOC)

# Doxygen looks into all activated module dirs
get_property(MODULE_DIRS GLOBAL PROPERTY MODULE_DIRS)
list(JOIN MODULE_DIRS " \\\n						 " KASPER_SOURCE_DIRS)

if( VTK_MAJOR_VERSION GREATER 5 )
    set( KASPER_PREDEFS VTK6 )
endif()

kasper_add_doc_reference(kasper.doxygen)

install(FILES ../README.md ../CHANGELOG.md ../CONTRIBUTING.md
        DESTINATION ${DOC_INSTALL_DIR})

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/index.html)
    install(FILES index.html DESTINATION ${DOC_INSTALL_DIR})
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../Changelog)
    install(DIRECTORY ../Changelog/
        DESTINATION ${DOC_INSTALL_DIR}/Changelog PATTERN "*.md")
endif()
