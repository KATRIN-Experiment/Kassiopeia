macro(set_path var value comment)
    if(NOT DEFINED ${var})
        set(${var} "${value}")
    else(NOT DEFINED ${var})
        set(${var} "${${var}}" CACHE PATH ${comment})
    endif(NOT DEFINED ${var})
endmacro(set_path)

include(CMakeDependentOption)
include(MacroParseArguments)

cmake_policy( SET CMP0011 NEW )

if( "${Kasper_SOURCE_DIR}" STREQUAL "${PROJECT_SOURCE_DIR}" )
    set(STANDALONE true)

    # use this section to modifiy initial values of builtin CMAKE variables
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    
        # modify the default installation prefix
        get_filename_component(BUILD_PARENT_DIR ${CMAKE_BINARY_DIR} PATH)
        set(CMAKE_INSTALL_PREFIX "${BUILD_PARENT_DIR}/install" CACHE PATH "Install path prefix, prepended onto install directories." FORCE)
    
        if(NOT CMAKE_BUILD_TYPE)
            set(CMAKE_BUILD_TYPE "" CACHE STRING "Choose build type (None | Debug | Release | RelWithDebInfo | MinSizeRel)" FORCE)
        endif()
        
        # set compiler warning levels
        set(_wflags_debug "-Wall -Wextra")
        set(_wflags_release "-Wall -Werror")
        
        string(STRIP "${CMAKE_CXX_FLAGS} ${_wflags_debug}" _cxx_flags)
        string(STRIP "${CMAKE_C_FLAGS} ${_wflags_debug}" _c_flags)
        
        set(CMAKE_CXX_FLAGS ${_cxx_flags} CACHE STRING "Flags used by the compiler during all build types." FORCE)
        set(CMAKE_C_FLAGS ${_c_flags} CACHE STRING "Flags used by the compiler during all build types." FORCE)
        
        #set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} ${_wflags_debug}" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        #set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${_wflags_debug}" CACHE STRING "Flags used by the compiler during debug builds." FORCE)
        #set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} ${_wflags_debug}" CACHE STRING "Flags used by the compiler during release with debug info builds." FORCE)
        #set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${_wflags_debug}" CACHE STRING "Flags used by the compiler during release with debug info builds." FORCE)
        
        #set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} ${_wflags_release}" CACHE STRING "Flags used by the compiler during min. size release builds." FORCE)
        #set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} ${_wflags_release}" CACHE STRING "Flags used by the compiler during min. size release builds." FORCE)
        #set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} ${_wflags_release}" CACHE STRING "Flags used by the compiler during release builds." FORCE)
        #set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${_wflags_release}" CACHE STRING "Flags used by the compiler during release builds." FORCE)
    endif()

    set_path(INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/include" "Install directory for headers")
    set_path(LIB_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/lib" "Install directory for libraries")
    set_path(BIN_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/bin" "Install directory for binaries")
    set_path(CONFIG_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/config" "Install directory for config files")
    set_path(DATA_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/data" "Install directory for data files")
    set_path(SCRATCH_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/scratch" "Directory for temporary files")
    set_path(OUTPUT_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/output" "Directory for output files")
    set_path(CACHE_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/cache" "Directory for cache files")
    set_path(LOG_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}/log" "Directory for log files")

    # a temporary fix to Apple's historical exclusion of system includes
    if ( APPLE AND NOT CMAKE_INCLUDE_SYSTEM_FLAG_CXX)
        set (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
    endif ()

    set( CMAKE_MACOSX_RPATH ON )

    # check compiler versions and set C++11 flag 
    include(CheckCompiler)
    message(STATUS "Using compiler ${COMPILER_ID} ${COMPILER_VERSION}")

    set(ROOT_FIND_QUIETLY TRUE)
    set(Sphinx_FIND_QUIETLY TRUE)
    set(Doxygen_FIND_QUIETLY TRUE)
    set(Boost_FIND_QUIETLY TRUE)
    set(VTK_FIND_QUIETLY TRUE)

    add_custom_target(reference)

else()
    set(STANDALONE false)

    # cleanup include dir definitions for this module
    set(MODULE_HEADER_DIRS)
    set(EXTERNAL_INCLUDE_DIRS)
endif()

set(CMAKE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/${PROJECT_NAME})
set(MODULE_INSTALL_DIR ${LIB_INSTALL_DIR}/cmake/modules)
file(RELATIVE_PATH REL_INCLUDE_INSTALL_DIR "${CMAKE_INSTALL_DIR}" "${INCLUDE_INSTALL_DIR}")
file(RELATIVE_PATH REL_MODULE_INSTALL_DIR "${CMAKE_INSTALL_DIR}" "${MODULE_INSTALL_DIR}")
set_property(GLOBAL PROPERTY MODULE_TARGETS)

find_package (Doxygen)
find_package (Sphinx)

###################
# path mechanisms #
###################

# mechanism for setting module-specific install paths
macro( kasper_module_paths PATH )

    set( ${PROJECT_NAME}_INCLUDE_INSTALL_DIR ${INCLUDE_INSTALL_DIR} )
    set( ${PROJECT_NAME}_LIB_INSTALL_DIR ${LIB_INSTALL_DIR} )
    set( ${PROJECT_NAME}_BIN_INSTALL_DIR ${BIN_INSTALL_DIR} )
    set( ${PROJECT_NAME}_CONFIG_INSTALL_DIR ${CONFIG_INSTALL_DIR}/${PATH} )
    set( ${PROJECT_NAME}_DATA_INSTALL_DIR ${DATA_INSTALL_DIR}/${PATH} )
    set( ${PROJECT_NAME}_SCRATCH_INSTALL_DIR ${SCRATCH_INSTALL_DIR}/${PATH} )
    set( ${PROJECT_NAME}_OUTPUT_INSTALL_DIR ${OUTPUT_INSTALL_DIR}/${PATH} )
    set( ${PROJECT_NAME}_LOG_INSTALL_DIR ${LOG_INSTALL_DIR}/${PATH} )
    set( ${PROJECT_NAME}_CACHE_INSTALL_DIR ${CACHE_INSTALL_DIR}/${PATH} )

    add_definitions( -DKASPER_INSTALL_DIR=${CMAKE_INSTALL_PREFIX} )

    add_definitions( -DINCLUDE_INSTALL_DIR=${${PROJECT_NAME}_INCLUDE_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_INCLUDE_INSTALL_DIR}\")" )
    add_definitions( -DLIB_INSTALL_DIR=${${PROJECT_NAME}_LIB_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_LIB_INSTALL_DIR}\")" )
    add_definitions( -DBIN_INSTALL_DIR=${${PROJECT_NAME}_BIN_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_BIN_INSTALL_DIR}\")" )
    add_definitions( -DCONFIG_INSTALL_DIR=${${PROJECT_NAME}_CONFIG_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_CONFIG_INSTALL_DIR}\")" )
    add_definitions( -DDATA_INSTALL_DIR=${${PROJECT_NAME}_DATA_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_DATA_INSTALL_DIR}\")" )
    add_definitions( -DSCRATCH_INSTALL_DIR=${${PROJECT_NAME}_SCRATCH_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_SCRATCH_INSTALL_DIR}\")" )
    add_definitions( -DOUTPUT_INSTALL_DIR=${${PROJECT_NAME}_OUTPUT_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_OUTPUT_INSTALL_DIR}\")" )
    add_definitions( -DLOG_INSTALL_DIR=${${PROJECT_NAME}_LOG_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_LOG_INSTALL_DIR}\")" )
    add_definitions( -DCACHE_INSTALL_DIR=${${PROJECT_NAME}_CACHE_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_CACHE_INSTALL_DIR}\")" )

endmacro()

####################
# debug mechanisms #
####################

# mechanism for building debug messages
macro( kasper_module_debug )
    # debug option for a module SECTION
    if (${ARGC} GREATER 0)
        set( ${PROJECT_NAME}_ENABLE_${ARGV0}_DEBUG OFF CACHE BOOL ${ARGV1} )
        if( ${PROJECT_NAME}_ENABLE_${ARGV0}_DEBUG )
            add_definitions( -D${PROJECT_NAME}_ENABLE_${ARGV0}_DEBUG )
        endif()
    # global module debug option
    else()
        set( ${PROJECT_NAME}_ENABLE_DEBUG OFF CACHE BOOL "build debug output for ${PROJECT_NAME}" )
        if( ${PROJECT_NAME}_ENABLE_DEBUG )
            add_definitions( -D${PROJECT_NAME}_ENABLE_DEBUG )
        endif()
    endif()
endmacro()

###################
# test mechanisms #
###################

# mechanism for building tests
macro( kasper_module_test TEST )
    set( ${PROJECT_NAME}_ENABLE_TEST OFF CACHE BOOL "build test programs for ${PROJECT_NAME}" )
    if( ${PROJECT_NAME}_ENABLE_TEST )
        add_definitions( -D${PROJECT_NAME}_ENABLE_TEST )
        add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/${TEST} )
    endif( ${PROJECT_NAME}_ENABLE_TEST )
endmacro()

#############
# utilities #
#############

macro(kasper_find_module NAME)
    if( ${ARGV1} )
        set( VERSION ${ARGV1} )
    else()
        set( VERSION "0" )
    endif()

    find_package( ${NAME} ${VERSION} REQUIRED NO_MODULE HINTS ${Kasper_BINARY_DIR}/${NAME} )
    #message("${NAME}_INCLUDE_DIRS: ${${NAME}_INCLUDE_DIRS}")
    kasper_internal_include_directories( ${${NAME}_INCLUDE_DIRS} )
    #kasper_external_include_directories( ${${NAME}_INCLUDE_DIRS} )
    #set( ${NAME}_INCLUDE_DIRS "${${NAME}_INCLUDE_DIRS}" PARENT_SCOPE )

endmacro()

macro(kasper_install_optional DEST_DIR)
    set(INST_CODE "")
    foreach(SRC_FILE ${ARGN})

        set(INST_CODE "
            ${INST_CODE}
            if(IS_ABSOLUTE \"${SRC_FILE}\")
                set( SRC_FILE_ABS \"${SRC_FILE}\" )
            else()
                find_file(SRC_FILE_ABS \"${SRC_FILE}\" HINTS \"${PROJECT_SOURCE_DIR}\" )
            endif()
            if(NOT EXISTS \${SRC_FILE_ABS})
                message(SEND_ERROR \"${SRC_FILE} could not be found.\")
            endif()
            get_filename_component(FILENAME \"\${SRC_FILE_ABS}\" NAME)
            set(DEST_FILE \"${DEST_DIR}/\${FILENAME}\")
            if(NOT EXISTS \"\${DEST_FILE}\")
                message(\"-- Installing: \${DEST_FILE}\" )
                configure_file(\"\${SRC_FILE_ABS}\" \"\${DEST_FILE}\" COPYONLY)
            else()
                file(READ \${SRC_FILE_ABS} SRC_CONTENT)
                file(READ \${DEST_FILE} DEST_CONTENT)
                if(SRC_CONTENT STREQUAL DEST_CONTENT)
                    message(\"-- Up-to-date: \${DEST_FILE}\" )
                else()
                    message(\"-- Installing: \${DEST_FILE}.dist\" )
                    configure_file(\"\${SRC_FILE_ABS}\" \"\${DEST_FILE}.dist\" COPYONLY)
                    if (\"\${SRC_FILE_ABS}\" IS_NEWER_THAN \"\${DEST_FILE}\")
                        message(\"** \${FILENAME} exists in a newer version.\" )
                    endif()
                endif()
            endif()
            unset(SRC_FILE_ABS CACHE)
        ")
    endforeach()
#    message("${INST_CODE}")
    if(NOT "${INST_CODE}" STREQUAL "")
        install(CODE "${INST_CODE}")
    endif()
endmacro()

macro(kasper_install_headers)
    install(FILES ${ARGN} DESTINATION ${${PROJECT_NAME}_INCLUDE_INSTALL_DIR})
endmacro()

macro(kasper_install_libraries)
    install(TARGETS ${ARGN} EXPORT KasperTargets DESTINATION ${${PROJECT_NAME}_LIB_INSTALL_DIR})
    set_property(GLOBAL APPEND PROPERTY MODULE_TARGETS ${ARGN})
    set_target_properties(${ARGN} PROPERTIES INSTALL_NAME_DIR ${${PROJECT_NAME}_LIB_INSTALL_DIR})
endmacro()

macro(kasper_install_executables)
    install(TARGETS ${ARGN} EXPORT KasperTargets DESTINATION ${${PROJECT_NAME}_BIN_INSTALL_DIR})
endmacro()

macro(kasper_install_data)
    install(FILES ${ARGN} DESTINATION ${${PROJECT_NAME}_DATA_INSTALL_DIR})
endmacro()

macro(kasper_configure_data)
    foreach(FILE ${ARGN})
        configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${FILE} DESTINATION ${${PROJECT_NAME}_DATA_INSTALL_DIR})
    endforeach()
endmacro()

macro(kasper_install_data_subdir SUBDIR)
    install(FILES ${ARGN} DESTINATION ${${PROJECT_NAME}_DATA_INSTALL_DIR}/${SUBDIR})
endmacro()

macro(kasper_configure_data_subdir SUBDIR)
    foreach(FILE ${ARGN})
        configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${FILE} DESTINATION ${${PROJECT_NAME}_DATA_INSTALL_DIR}/${SUBDIR})
    endforeach()
endmacro()

macro(kasper_install_config)
    install(FILES ${ARGN} DESTINATION ${${PROJECT_NAME}_CONFIG_INSTALL_DIR})
endmacro()

macro(kasper_configure_config)
    foreach(FILE ${ARGN})
        configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${FILE} DESTINATION ${${PROJECT_NAME}_CONFIG_INSTALL_DIR})
    endforeach()
endmacro()

macro(kasper_install_config_subdir SUBDIR)
    install(FILES ${ARGN} DESTINATION ${${PROJECT_NAME}_CONFIG_INSTALL_DIR}/${SUBDIR})
endmacro()

macro(kasper_configure_config_subdir SUBDIR)
    foreach(FILE ${ARGN})
        configure_file(${FILE} ${CMAKE_CURRENT_BINARY_DIR}/${FILE} @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${FILE} DESTINATION ${${PROJECT_NAME}_CONFIG_INSTALL_DIR}/${SUBDIR})
    endforeach()
endmacro()

macro(kasper_install_files DEST_DIR)
    install(FILES ${ARGN} DESTINATION ${DEST_DIR})
endmacro()


macro(kasper_install_module)

    configure_file(ModuleConfigVersion.cmake.in ${PROJECT_NAME}ConfigVersion.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake DESTINATION ${CMAKE_INSTALL_DIR})

    get_property(MODULE_TARGETS GLOBAL PROPERTY MODULE_TARGETS)
    #list(APPEND EXTERNAL_INCLUDE_DIRS ${KASPER_INCLUDE_DIRS})

    set(INSTALLED_INCLUDE_DIRS ${INCLUDE_INSTALL_DIR})

    foreach(DIR ${EXTERNAL_INCLUDE_DIRS})
        if (NOT ${DIR} STREQUAL "SYSTEM")
            file(RELATIVE_PATH REL_DIR ${CMAKE_SOURCE_DIR} ${DIR})
            if(REL_DIR MATCHES "^\\.\\.")
                list(APPEND INSTALLED_INCLUDE_DIRS ${DIR})
            endif()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES INSTALLED_INCLUDE_DIRS)

    configure_file(ModuleConfigInstalled.cmake.in ${PROJECT_NAME}ConfigInstalled.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigInstalled.cmake DESTINATION ${CMAKE_INSTALL_DIR}
            RENAME ${PROJECT_NAME}Config.cmake)

    if(STANDALONE)
        install(EXPORT KasperTargets DESTINATION ${CMAKE_INSTALL_DIR} FILE ModuleTargets.cmake)
    else()

        set(MODULE_INCLUDE_DIRS )
        set(TMP_DIR_LIST ${MODULE_HEADER_DIRS};${EXTERNAL_INCLUDE_DIRS})
        foreach(DIR ${TMP_DIR_LIST})
            string(REPLACE " " "\\\ " DIR ${DIR})
            list(APPEND MODULE_INCLUDE_DIRS ${DIR})
        endforeach()

        list( LENGTH MODULE_INCLUDE_DIRS LL)
        if( LL GREATER 1 )
            list( REMOVE_DUPLICATES MODULE_INCLUDE_DIRS )
        endif()

        configure_file(ModuleConfig.cmake.in ${PROJECT_NAME}Config.cmake @ONLY)
    endif()

    #pkg-config
    include(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake)

    set(DESCRIPTION ${ARGV0})

    set( PC_LIBRARIES_STR )
    set( LIBDIRS )
    set( LIBS ${${PROJECT_NAME}_LIBRARIES} ${DEPENDS_LIBRARIES} )
    foreach(LIB ${LIBS})

        # try to match a directory name
        string(REGEX MATCH "^.*/" LIBDIR ${LIB})

        if (LIBDIR)
            string(LENGTH ${LIBDIR} LIBDIRLEN)
            math(EXPR LIBDIRLEN "${LIBDIRLEN}-1")
            string(SUBSTRING ${LIBDIR} 0 ${LIBDIRLEN} LIBDIR)
            #message("${LIBDIR}")
            list(FIND LIBDIRS ${LIBDIR} LIBDIRINDEX)
            if (LIBDIRINDEX LESS 0)

                string(REGEX MATCH "^/usr" SYSLIBDIR ${LIBDIR})
                string(REGEX MATCH "^/usr.*(root|boost)" SPECIALSYSLIBDIR ${LIBDIR})
                if(NOT SYSLIBDIR)
                    list(INSERT LIBDIRS 0 ${LIBDIR})
                elseif(SPECIALSYSLIBDIR)
                    list(APPEND LIBDIRS ${LIBDIR})
                endif()

            endif()
            STRING(REGEX REPLACE "${LIBDIR}/?(lib)?([^\\.]+)(\\..+)?$" "-l\\2" LIB ${LIB})
            set ( PC_LIBRARIES_STR "${PC_LIBRARIES_STR} ${LIB}" )
        elseif(${LIB} MATCHES "^\\-" OR ${LIB} MATCHES "\\.")
            set ( PC_LIBRARIES_STR "${PC_LIBRARIES_STR} ${LIB}" )
        else()
            set ( PC_LIBRARIES_STR "${PC_LIBRARIES_STR} -l${LIB}${SHARED_LIBRARY_SUFFIX}" )
        endif()
    endforeach()

    set( PC_LD_FLAGS "-L\${libdir}" )
    set( PC_RPATH_STR "\${libdir}" )
    foreach(LIBDIR ${LIBDIRS})
        set ( PC_LD_FLAGS "${PC_LD_FLAGS} -L${LIBDIR}" )
        set ( PC_RPATH_STR "${PC_RPATH_STR} ${LIBDIR}" )
    endforeach()

    set( PC_LIBRARIES_STR "${PC_LD_FLAGS} ${PC_LIBRARIES_STR}" )

    set( PC_INCLUDE_DIR_STR "-I\${includedir}" )
    foreach(DIR ${INSTALLED_INCLUDE_DIRS})
        if(NOT DIR STREQUAL INCLUDE_INSTALL_DIR)
            set ( PC_INCLUDE_DIR_STR "${PC_INCLUDE_DIR_STR} -I${DIR}" )
        endif()
    endforeach()

    set( PC_CONTENTS "prefix=${CMAKE_INSTALL_PREFIX}
exec_prefix=${BIN_INSTALL_DIR}
libdir=${LIB_INSTALL_DIR}
includedir=${INCLUDE_INSTALL_DIR}
rpath=${PC_RPATH_STR}

Name: ${PROJECT_NAME}
Description: ${DESCRIPTION}
Version: ${MODULE_VERSION}

Libs: ${PC_LIBRARIES_STR}
Cflags: ${CXX11_FLAG} ${PC_INCLUDE_DIR_STR}
")
    string(TOLOWER ${PROJECT_NAME} PROJECT_NAME_LOWER )
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME_LOWER}.pc ${PC_CONTENTS})
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME_LOWER}.pc DESTINATION ${LIB_INSTALL_DIR}/pkgconfig )
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME_LOWER}.pc DESTINATION ${LIB_INSTALL_DIR}/pkgconfig
            RENAME ${PROJECT_NAME}.pc )

endmacro()

macro(kasper_add_doc_reference DOXYGEN_FILE)
    # Builds in the source tree (to instead build in the binary tree, switch CMAKE_CURRENT_SOURCE_DIR to CMAKE_CURRENT_BINARY_DIR in the line that starts with WORKING_DIRECTORY.
    # This is intended for use by developers only when updating the reference documentation.
    if (DOXYGEN_FOUND)
        if(NOT PACKAGE_VERSION)
            set(PACKAGE_VERSION ${MODULE_VERSION_MAJOR})
        endif()
        configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Reference/${DOXYGEN_FILE}.in ${CMAKE_CURRENT_BINARY_DIR}/Reference/${DOXYGEN_FILE} @ONLY)
        set(REF_BUILD_DIR ${CMAKE_INSTALL_PREFIX}/doc/${PROJECT_NAME}/Reference)
        file(MAKE_DIRECTORY ${REF_BUILD_DIR})
        add_custom_target (reference-${PROJECT_NAME}
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Reference/${DOXYGEN_FILE}
            WORKING_DIRECTORY ${REF_BUILD_DIR}
            COMMENT "Generating API documentation with Doxygen for ${PROJECT_NAME}"
            VERBATIM
        )
        add_dependencies(reference reference-${PROJECT_NAME})
    endif(DOXYGEN_FOUND)
endmacro()

macro(kasper_add_doc_reference_sphinx SPHINX_FILE)
    # Builds in the source tree (to instead build in the binary tree, switch CMAKE_CURRENT_SOURCE_DIR to CMAKE_CURRENT_BINARY_DIR in the line that starts with WORKING_DIRECTORY.
    # This is intended for use by developers only when updating the reference documentation.
    if (SPHINX_FOUND)
        if(NOT PACKAGE_VERSION)
            set(PACKAGE_VERSION ${MODULE_VERSION_MAJOR})
        endif()
        configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Reference/conf.py.in ${CMAKE_CURRENT_BINARY_DIR}/Reference/conf.py @ONLY)
        configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Reference/index.rst.in ${CMAKE_CURRENT_BINARY_DIR}/Reference/index.rst @ONLY)
        configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Reference/${SPHINX_FILE}.in ${CMAKE_CURRENT_BINARY_DIR}/Reference/${SPHINX_FILE} @ONLY)
        set(REF_BUILD_DIR ${CMAKE_INSTALL_PREFIX}/doc/${PROJECT_NAME}/Reference)
        file(MAKE_DIRECTORY ${REF_BUILD_DIR})
        add_custom_target (reference-${PROJECT_NAME}
            ${SPHINX_EXECUTABLE} -b html ${CMAKE_CURRENT_BINARY_DIR}/Reference/ ${REF_BUILD_DIR}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Reference/
            COMMENT "Generating API documentation with Sphinx for ${PROJECT_NAME}"
            VERBATIM
        )
        add_dependencies(reference reference-${PROJECT_NAME})
    endif(SPHINX_FOUND)
endmacro()

macro(kasper_install_doc)
    set(DOC_INSTALL_DIR ${CMAKE_INSTALL_PREFIX}/doc/${PROJECT_NAME})
    install(FILES ${ARGN} DESTINATION ${DOC_INSTALL_DIR})
endmacro()


# macros to facilitate a modular build via subdirectories

macro( kasper_include_directories )

    set(KASPER_DIRS)
    set(SYSTEM_DIRS)

    foreach(dir ${ARGN})
        string(FIND ${dir} ${CMAKE_SOURCE_DIR} pos)
        if(pos LESS 0)
            list(APPEND SYSTEM_DIRS ${dir})
        else()
            list(APPEND KASPER_DIRS ${dir})
        endif()
    endforeach()

    include_directories( ${KASPER_DIRS} )
    include_directories( SYSTEM;${SYSTEM_DIRS} )
endmacro()

macro( kasper_include_default_dirs )
    kasper_include_directories( ${MODULE_HEADER_DIRS};${EXTERNAL_INCLUDE_DIRS} )
endmacro()

macro( kasper_internal_include_directories )

    foreach(dir ${ARGN})
        #message("${dir}")
        if(NOT IS_ABSOLUTE ${dir})
            set(dir ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
        endif()
        list(APPEND MODULE_HEADER_DIRS ${dir})
    endforeach()

    list( LENGTH MODULE_HEADER_DIRS LL)
    if( LL GREATER 1 )
        list( REMOVE_DUPLICATES MODULE_HEADER_DIRS )
    endif()

    if (CMAKE_CURRENT_SOURCE_DIR STRGREATER PROJECT_SOURCE_DIR)
        set (MODULE_HEADER_DIRS ${MODULE_HEADER_DIRS} PARENT_SCOPE)
    endif()

    kasper_include_directories( ${MODULE_HEADER_DIRS} )

endmacro()


macro( kasper_external_include_directories )

    list( APPEND EXTERNAL_INCLUDE_DIRS ${ARGN} )

    list( LENGTH EXTERNAL_INCLUDE_DIRS LL)
    if( LL GREATER 1 )
        list( REMOVE_ITEM EXTERNAL_INCLUDE_DIRS SYSTEM )
        list( REMOVE_DUPLICATES EXTERNAL_INCLUDE_DIRS )
    endif()

    if (CMAKE_CURRENT_SOURCE_DIR STRGREATER PROJECT_SOURCE_DIR)
        #message("CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR} -- PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
        set( EXTERNAL_INCLUDE_DIRS ${EXTERNAL_INCLUDE_DIRS} PARENT_SCOPE )
    endif()

    kasper_include_directories( ${ARGN} )

endmacro()

macro (add_cflag CFLAG)
    list (APPEND MODULE_CFLAGS ${CFLAG})
    set (MODULE_CFLAGS ${MODULE_CFLAGS} PARENT_SCOPE)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D${CFLAG}")
    set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} PARENT_SCOPE)
endmacro()

macro(kasper_find_boost BVERSION)
    CMAKE_PARSE_ARGUMENTS(ARG "REQUIRED" "" "COMPONENTS" ${ARGN})
    if(ARG_REQUIRED)
        set(ARG_REQUIRED "REQUIRED")
    else()
        set(ARG_REQUIRED )
    endif()
    
    find_package(Boost ${BVERSION} ${ARG_REQUIRED} COMPONENTS ${ARG_COMPONENTS})
    
    # circumvent a C++11 boost bug in the filesystem library
    if (Boost_VERSION LESS 105100)
        add_definitions( -DBOOST_NO_SCOPED_ENUMS )
    elseif (Boost_VERSION LESS 105700)
        add_definitions( -DBOOST_NO_CXX11_SCOPED_ENUMS )
    endif()
    
    kasper_external_include_directories( ${Boost_INCLUDE_DIRS} )
endmacro()

macro(kasper_find_vtk)                                   
    # VTK versions below 6.0.1 do not compile with c++11 support
    find_package( VTK REQUIRED NO_MODULE )
    include(${VTK_USE_FILE})

    if(${VTK_VERSION} VERSION_LESS "6.0.1")
        message(FATAL_ERROR "At least VTK version 6.0.1 is required for C++11 support.")
    endif()

    if(VTK_VERSION VERSION_GREATER "6" AND VTK_QT_VERSION VERSION_GREATER "4")
        find_package(Qt5Widgets REQUIRED QUIET)
    endif()

    kasper_external_include_directories( ${VTK_INCLUDE_DIRS} )
endmacro()
