cmake_policy(SET CMP0048 NEW)

macro(set_path var value comment)
    if(NOT DEFINED ${var})
        set(${var} "${value}")
    else(NOT DEFINED ${var})
        set(${var} "${${var}}" CACHE PATH ${comment})
    endif(NOT DEFINED ${var})
endmacro(set_path)

include(CMakeDependentOption)
include(MacroParseArguments)
include(GNUInstallDirs)

if( ${CMAKE_SOURCE_DIR} STREQUAL ${PROJECT_SOURCE_DIR} )

    set(STANDALONE true)

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/CMakeCache.txt")
        message(FATAL_ERROR "Please remove the file '${CMAKE_CURRENT_SOURCE_DIR}/CMakeCache.txt' before running `cmake`.")
    endif()
    if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}" OR EXISTS "${CMAKE_CURRENT_BINARY_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Please run the `cmake` command from the build directory, not inside the source tree! See the file 'README.md' for instructions.")
    endif()

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

    # define global install paths
    set_path(KASPER_INSTALL_DIR "${CMAKE_INSTALL_PREFIX}" "Kasper install directory")
    set_path(INCLUDE_INSTALL_DIR "${KASPER_INSTALL_DIR}/${CMAKE_INSTALL_INCLUDEDIR}" "Install directory for headers")
    set_path(LIB_INSTALL_DIR "${KASPER_INSTALL_DIR}/${CMAKE_INSTALL_LIBDIR}" "Install directory for libraries")
    set_path(BIN_INSTALL_DIR "${KASPER_INSTALL_DIR}/${CMAKE_INSTALL_BINDIR}" "Install directory for binaries")
    set_path(DOC_INSTALL_DIR "${KASPER_INSTALL_DIR}/doc" "Install directory for documentation files")
    set_path(CONFIG_INSTALL_DIR "${KASPER_INSTALL_DIR}/config" "Install directory for config files")
    set_path(DATA_INSTALL_DIR "${KASPER_INSTALL_DIR}/data" "Install directory for data files")
    set_path(SCRATCH_INSTALL_DIR "${KASPER_INSTALL_DIR}/scratch" "Directory for temporary files")
    set_path(OUTPUT_INSTALL_DIR "${KASPER_INSTALL_DIR}/output" "Directory for output files")
    set_path(CACHE_INSTALL_DIR "${KASPER_INSTALL_DIR}/cache" "Directory for cache files")
    set_path(LOG_INSTALL_DIR "${KASPER_INSTALL_DIR}/log" "Directory for log files")
    set_path(CMAKE_INSTALL_DIR "${LIB_INSTALL_DIR}/cmake" "Directory for CMake files" )
    set_path(MODULE_INSTALL_DIR "${LIB_INSTALL_DIR}/cmake/modules" "Directory for CMake module files")

    message(STATUS "*** Kasper install path is: ${KASPER_INSTALL_DIR} [${CMAKE_INSTALL_LIBDIR}]")

    # remove old bin/lib install files to avoid conflicts - do NOT touch config/log/scratch/...
    install(CODE "file(REMOVE_RECURSE ${LIB_INSTALL_DIR} ${BIN_INSTALL_DIR})")

    # a temporary fix to Apple's historical exclusion of system includes
    if ( APPLE AND NOT CMAKE_INCLUDE_SYSTEM_FLAG_CXX)
        set (CMAKE_INCLUDE_SYSTEM_FLAG_CXX "-isystem ")
    endif ()

    set( CMAKE_MACOSX_RPATH ON )

    # check compiler versions and set C++11 flag
    include(CheckCompiler)
    message(STATUS "Using compiler ${COMPILER_ID} ${COMPILER_VERSION}")

    set(ROOT_FIND_QUIETLY TRUE)
    set(HDF5_FIND_QUIETLY TRUE)
    set(Sphinx_FIND_QUIETLY TRUE)
    set(Doxygen_FIND_QUIETLY TRUE)
    set(Boost_FIND_QUIETLY TRUE)
    set(VTK_FIND_QUIETLY TRUE)

    add_custom_target(doc)        # full docs
    add_custom_target(reference)  # submodule docs
else()
    set(STANDALONE false)

    set_property(GLOBAL PROPERTY ${PROJECT_NAME}_LIBRARIES "")
endif()

macro(kasper_set_version_numbers VERSION_VAR)
    if( NOT ${VERSION_VAR}_MINOR )
        set(${VERSION_VAR}_MINOR 0)
    endif()
    if( NOT ${VERSION_VAR}_PATCH )
        set(${VERSION_VAR}_PATCH 0)
    endif()
    if( NOT ${VERSION_VAR})
        set( ${VERSION_VAR} "${${VERSION_VAR}_MAJOR}.${${VERSION_VAR}_MINOR}.${${VERSION_VAR}_PATCH}" )
    endif()

    if( (${${VERSION_VAR}_MINOR} GREATER 99) OR (${${VERSION_VAR}_PATCH} GREATER 99) )
        message(FATAL_ERROR "Invalid version number defined for project ${PROJECT_NAME}!")
    endif()
    math( EXPR ${VERSION_VAR}_NUMERICAL "(${${VERSION_VAR}_MAJOR} * 10000) + (${${VERSION_VAR}_MINOR} * 100) + ${${VERSION_VAR}_PATCH}" )

    if(NOT PROJECT_VERSION)
        set( PROJECT_VERSION_MAJOR ${${VERSION_VAR}_MAJOR} )
        set( PROJECT_VERSION_MINOR ${${VERSION_VAR}_MINOR} )
        set( PROJECT_VERSION_PATCH ${${VERSION_VAR}_PATCH} )
        set( PROJECT_VERSION ${${VERSION_VAR}} )
    endif()

    set( ${PROJECT_NAME}_VERSION_MAJOR ${${VERSION_VAR}_MAJOR} )
    set( ${PROJECT_NAME}_VERSION_MINOR ${${VERSION_VAR}_MINOR} )
    set( ${PROJECT_NAME}_VERSION_PATCH ${${VERSION_VAR}_PATCH} )
    set( ${PROJECT_NAME}_VERSION_NUMERICAL ${${VERSION_VAR}_NUMERICAL} )
    set( ${PROJECT_NAME}_VERSION ${${VERSION_VAR}} )

    add_compile_definitions( ${PROJECT_NAME}_VERSION_MAJOR=${${PROJECT_NAME}_VERSION_MAJOR} )
    add_compile_definitions( ${PROJECT_NAME}_VERSION_MINOR=${${PROJECT_NAME}_VERSION_MINOR} )
    add_compile_definitions( ${PROJECT_NAME}_VERSION_PATCH=${${PROJECT_NAME}_VERSION_PATCH} )
    add_compile_definitions( ${PROJECT_NAME}_VERSION_NUMERICAL=${${PROJECT_NAME}_VERSION_NUMERICAL} )
    add_compile_definitions( ${PROJECT_NAME}_VERSION="${${PROJECT_NAME}_VERSION}" )
endmacro()

if( ${PROJECT_NAME} STREQUAL Kasper )
    kasper_set_version_numbers(KASPER_VERSION)
    message(STATUS "Kasper version is v${KASPER_VERSION} (${KASPER_VERSION_NUMERICAL})" )

    # git revision (if available)
    set(KASPER_GIT_REVISION "n/a")
    if(EXISTS "${CMAKE_SOURCE_DIR}/.git/index")
        set_property(GLOBAL APPEND
            PROPERTY CMAKE_CONFIGURE_DEPENDS
            "${CMAKE_SOURCE_DIR}/.git/index")

        execute_process(
            COMMAND git rev-parse --abbrev-ref HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE KASPER_GIT_BRANCH
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND git rev-parse --short HEAD
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE KASPER_GIT_COMMIT
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
            COMMAND git log -1 --format=%cd
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE KASPER_GIT_TIMESTAMP
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(KASPER_GIT_REVISION "${KASPER_GIT_BRANCH}+${KASPER_GIT_COMMIT}")
        message(STATUS "Git revision is ${KASPER_GIT_REVISION} (last commit: ${KASPER_GIT_TIMESTAMP})" )
    endif()

    # build timestamp -- will be refreshed after updating git (see lines above)
    string( TIMESTAMP KASPER_BUILD_TIMESTAMP UTC )

    # build system (something like 'linux/GNU/8.2.1')
    set(KASPER_BUILD_SYSTEM "${CMAKE_SYSTEM_NAME}/${CMAKE_CXX_COMPILER_ID}/${CMAKE_CXX_COMPILER_VERSION}")

else()
    kasper_set_version_numbers(MODULE_VERSION)
    message(STATUS "Kasper module enabled: ${PROJECT_NAME} v${${PROJECT_NAME}_VERSION} (${${PROJECT_NAME}_VERSION_NUMERICAL})" )
endif()

find_package (Doxygen)
find_package (Sphinx)

###################
# path mechanisms #
###################

# mechanism for setting module-specific install paths
macro( kasper_module_paths PATH )

    set( ${PROJECT_NAME}_INCLUDE_INSTALL_DIR "${INCLUDE_INSTALL_DIR}" )
    set( ${PROJECT_NAME}_LIB_INSTALL_DIR "${LIB_INSTALL_DIR}" )
    set( ${PROJECT_NAME}_BIN_INSTALL_DIR "${BIN_INSTALL_DIR}" )
    #set( ${PROJECT_NAME}_DOC_INSTALL_DIR "${DOC_INSTALL_DIR}/${PATH}" )
    set( ${PROJECT_NAME}_CONFIG_INSTALL_DIR "${CONFIG_INSTALL_DIR}/${PATH}" )
    set( ${PROJECT_NAME}_DATA_INSTALL_DIR "${DATA_INSTALL_DIR}/${PATH}" )
    set( ${PROJECT_NAME}_SCRATCH_INSTALL_DIR "${SCRATCH_INSTALL_DIR}/${PATH}" )
    set( ${PROJECT_NAME}_OUTPUT_INSTALL_DIR "${OUTPUT_INSTALL_DIR}/${PATH}" )
    set( ${PROJECT_NAME}_LOG_INSTALL_DIR "${LOG_INSTALL_DIR}/${PATH}" )
    set( ${PROJECT_NAME}_CACHE_INSTALL_DIR "${CACHE_INSTALL_DIR}/${PATH}" )

    add_compile_definitions( KASPER_INSTALL_DIR=${KASPER_INSTALL_DIR} )

    add_compile_definitions( INCLUDE_INSTALL_DIR=${${PROJECT_NAME}_INCLUDE_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_INCLUDE_INSTALL_DIR}\")" )
    add_compile_definitions( LIB_INSTALL_DIR=${${PROJECT_NAME}_LIB_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_LIB_INSTALL_DIR}\")" )
    add_compile_definitions( BIN_INSTALL_DIR=${${PROJECT_NAME}_BIN_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_BIN_INSTALL_DIR}\")" )
    #add_compile_definitions( DOC_INSTALL_DIR=${${PROJECT_NAME}_DOC_INSTALL_DIR} )
    #install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_DOC_INSTALL_DIR}\")" )
    add_compile_definitions( CONFIG_INSTALL_DIR=${${PROJECT_NAME}_CONFIG_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_CONFIG_INSTALL_DIR}\")" )
    add_compile_definitions( DATA_INSTALL_DIR=${${PROJECT_NAME}_DATA_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_DATA_INSTALL_DIR}\")" )
    add_compile_definitions( SCRATCH_INSTALL_DIR=${${PROJECT_NAME}_SCRATCH_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_SCRATCH_INSTALL_DIR}\")" )
    add_compile_definitions( OUTPUT_INSTALL_DIR=${${PROJECT_NAME}_OUTPUT_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_OUTPUT_INSTALL_DIR}\")" )
    add_compile_definitions( LOG_INSTALL_DIR=${${PROJECT_NAME}_LOG_INSTALL_DIR} )
    install(CODE "file(MAKE_DIRECTORY \"${${PROJECT_NAME}_LOG_INSTALL_DIR}\")" )
    add_compile_definitions( CACHE_INSTALL_DIR=${${PROJECT_NAME}_CACHE_INSTALL_DIR} )
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
            add_compile_definitions( ${PROJECT_NAME}_ENABLE_${ARGV0}_DEBUG )
        endif()
    # global module debug option
    else()
        set( ${PROJECT_NAME}_ENABLE_DEBUG OFF CACHE BOOL "build debug output for ${PROJECT_NAME}" )
        if( ${PROJECT_NAME}_ENABLE_DEBUG )
            add_compile_definitions( ${PROJECT_NAME}_ENABLE_DEBUG )
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
        add_compile_definitions( ${PROJECT_NAME}_ENABLE_TEST )
        add_subdirectory( ${CMAKE_CURRENT_SOURCE_DIR}/${TEST} )
    endif( ${PROJECT_NAME}_ENABLE_TEST )
endmacro()

#############
# utilities #
#############

macro(kasper_append_paths PARENT_VAR)
    foreach(RELPATH ${ARGN})
        get_filename_component(ABSPATH ${RELPATH} ABSOLUTE)
        list(APPEND ${PARENT_VAR} ${ABSPATH})
    endforeach()
    set (${PARENT_VAR} ${${PARENT_VAR}} PARENT_SCOPE)
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

macro(kasper_install_header_directory)
    install(DIRECTORY ${ARGN} DESTINATION ${${PROJECT_NAME}_INCLUDE_INSTALL_DIR})
endmacro()

macro(kasper_install_libraries)
    install(TARGETS  ${ARGN}
        EXPORT ${PROJECT_NAME}Targets
        DESTINATION ${${PROJECT_NAME}_LIB_INSTALL_DIR}
    )

    foreach(ARG ${ARGN})
        get_target_property(TARGET_TYPE ${ARG} TYPE)
        if(NOT ${TARGET_TYPE} STREQUAL INTERFACE_LIBRARY)
            set_target_properties(${ARG} PROPERTIES INSTALL_NAME_DIR ${${PROJECT_NAME}_LIB_INSTALL_DIR})
        endif()

        # append to list of project librariers
        # TODO: there should be a smarter method to get this directly from the CMake export set...
        set_property(GLOBAL APPEND PROPERTY ${PROJECT_NAME}_LIBRARIES ${ARG})
    endforeach()
endmacro()

macro(kasper_install_executables)
    install(TARGETS  ${ARGN}
        EXPORT ${PROJECT_NAME}Targets
        DESTINATION ${${PROJECT_NAME}_BIN_INSTALL_DIR}
    )
endmacro()

macro(kasper_install_executables_with_symlinks LINK_NAME)
    kasper_install_executables(${ARGN})
    foreach( EXECUTABLE_NAME ${ARGN} )
        set(LINK_TARGET_NAME ${LINK_NAME}${EXECUTABLE_NAME})
        install(CODE "execute_process( \
            COMMAND ${CMAKE_COMMAND} -E create_symlink \
                ${${PROJECT_NAME}_BIN_INSTALL_DIR}/${EXECUTABLE_NAME} \
                ${${PROJECT_NAME}_BIN_INSTALL_DIR}/${LINK_TARGET_NAME} \
            )"
        )
    endforeach( EXECUTABLE_NAME )
endmacro()

macro(kasper_install_script)
    install(
        FILES ${ARGN}
        PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
        DESTINATION ${${PROJECT_NAME}_BIN_INSTALL_DIR}
    )
endmacro()

macro(kasper_install_data)
    install(FILES ${ARGN} DESTINATION ${${PROJECT_NAME}_DATA_INSTALL_DIR})
endmacro()

macro(kasper_install_config)
    install(FILES ${ARGN} DESTINATION ${${PROJECT_NAME}_CONFIG_INSTALL_DIR})
endmacro()

macro(kasper_install_config_subdir SUBDIR)
    install(FILES ${ARGN} DESTINATION ${${PROJECT_NAME}_CONFIG_INSTALL_DIR}/${SUBDIR})
endmacro()

macro(kasper_install_files DEST_DIR)
    install(FILES ${ARGN} DESTINATION ${DEST_DIR})
endmacro()

macro(kasper_install_module)
    kasper_set_version_numbers(MODULE_VERSION)
    #message(STATUS "Processed module: ${PROJECT_NAME} v${PROJECT_VERSION}")

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

    configure_file(${PROJECT_NAME}ConfigVersion.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
            DESTINATION ${CMAKE_INSTALL_DIR}/${PROJECT_NAME}
    )

    install(EXPORT ${PROJECT_NAME}Targets
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION ${CMAKE_INSTALL_DIR}/${PROJECT_NAME}
    )
    configure_file(${PROJECT_NAME}Config.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
        DESTINATION ${CMAKE_INSTALL_DIR}/${PROJECT_NAME}
    )

    #pkg-config
    kasper_export_pkgconfig()
endmacro()

macro(kasper_export_pkgconfig)

    # get the libraries associated with the project
    get_property(${PROJECT_NAME}_LIBRARIES GLOBAL PROPERTY ${PROJECT_NAME}_LIBRARIES)

    set(${PROJECT_NAME}_DEPENDS)
    set(${PROJECT_NAME}_CXX_FLAGS)

    # generate list of dependencies for installed libraries for use in the pkgconfig file
    foreach(TARGET_NAME ${${PROJECT_NAME}_LIBRARIES})
        get_target_property(TARGET_LIBS ${TARGET_NAME} INTERFACE_LINK_LIBRARIES)

        # scan each dependency of the current target
        foreach(LIB_NAME ${TARGET_NAME} ${TARGET_LIBS})
            if(TARGET ${LIB_NAME})
                get_target_property(LIB_TYPE ${LIB_NAME} TYPE)
                # ignore interface libs that cannot be used by external software
                if(NOT LIB_TYPE STREQUAL INTERFACE_LIBRARY)
                    get_target_property(LIB_IMPORTED ${LIB_NAME} IMPORTED)
                    if(LIB_IMPORTED)
                        # get the actual filename on disk if it exists
                        get_target_property(LIB_LOCATION ${LIB_NAME} LOCATION)
                        if(LIB_LOCATION)
                            list(APPEND ${PROJECT_NAME}_DEPENDS ${LIB_LOCATION})
                        endif(LIB_LOCATION)
                    else(LIB_IMPORTED)
                        list(APPEND ${PROJECT_NAME}_DEPENDS ${LIB_NAME})
                    endif(LIB_IMPORTED)
                endif()
            endif()
        endforeach(LIB_NAME)

        get_target_property(TARGET_LINK_OPTS ${TARGET_NAME} INTERFACE_LINK_OPTIONS)
        if(TARGET_LINK_OPTS)
            foreach(FLAG ${TARGET_LINK_OPTS})
                list(APPEND ${PROJECT_NAME}_DEPENDS ${FLAG})
            endforeach(FLAG)
        endif()

        get_target_property(TARGET_OPTS ${TARGET_NAME} INTERFACE_COMPILE_OPTIONS)
        if(TARGET_OPTS)
            foreach(FLAG ${TARGET_OPTS})
                if(NOT ${FLAG} MATCHES "-W")
                    list(APPEND ${PROJECT_NAME}_CXX_FLAGS ${FLAG})
                endif()
            endforeach(FLAG)
        endif(TARGET_OPTS)

        get_target_property(TARGET_DEFS ${TARGET_NAME} INTERFACE_COMPILE_DEFINITIONS)
        if(TARGET_DEFS)
            foreach(FLAG ${TARGET_DEFS})
                list(APPEND ${PROJECT_NAME}_CXX_FLAGS -D${FLAG})
            endforeach(FLAG)
        endif(TARGET_DEFS)

        get_target_property(TARGET_INCLUDE_DIRS ${TARGET_NAME} INTERFACE_INCLUDE_DIRECTORIES)
        if(TARGET_INCLUDE_DIRS)
            foreach(INC_DIR ${TARGET_INCLUDE_DIRS})
                # ignore cmake generator expressions
                if(NOT (${INC_DIR} MATCHES "BUILD_INTERFACE:" OR ${INC_DIR} MATCHES "INSTALL_INTERFACE:"))
                    list(APPEND ${PROJECT_NAME}_CXX_FLAGS -I${INC_DIR})
                endif()
            endforeach(INC_DIR)
        endif(TARGET_INCLUDE_DIRS)

    endforeach(TARGET_NAME)

    #message("${PROJECT_NAME}_CXX_FLAGS : ${${PROJECT_NAME}_CXX_FLAGS}")
    #message("${PROJECT_NAME}_LIBRARIES : ${${PROJECT_NAME}_LIBRARIES}")
    #message("${PROJECT_NAME}_DEPENDS   : ${${PROJECT_NAME}_DEPENDS}")

    #include(${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake)

    set(DESCRIPTION ${ARGV0})

    set( PC_LIBRARIES_STR )
    set( LIBDIRS )
    #set( LIBS ${${PROJECT_NAME}_LIBRARIES} ${${PROJECT_NAME}_DEPENDS} )
    set( LIBS ${${PROJECT_NAME}_DEPENDS} )
    if(LIBS)
        list( REMOVE_DUPLICATES LIBS )
    endif(LIBS)

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
                string(REGEX MATCH "^/usr.*(root|boost|vtk)" SPECIALSYSLIBDIR ${LIBDIR})
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

    set( PC_LD_FLAGS_STR "-L\${libdir}" )
    #if(GCC_FORCE_LINKING)
    #    set( PC_LD_FLAGS_STR "${PC_LD_FLAGS_STR} -Wl,--no-as-needed" )
    #endif(GCC_FORCE_LINKING)

    set( PC_RPATH_STR "\${libdir}" )
    foreach(LIBDIR ${LIBDIRS})
        set ( PC_LD_FLAGS_STR "${PC_LD_FLAGS_STR} -L${LIBDIR}" )
        set ( PC_RPATH_STR "${PC_RPATH_STR} ${LIBDIR}" )
    endforeach(LIBDIR)

    set( PC_INCLUDE_DIR_STR "-I\${includedir}" )
    foreach(DIR ${INSTALLED_INCLUDE_DIRS})
        if(NOT DIR STREQUAL INCLUDE_INSTALL_DIR)
            set ( PC_INCLUDE_DIR_STR "${PC_INCLUDE_DIR_STR} -I${DIR}" )
        endif()
    endforeach(DIR)

    set( PC_CXX_FLAGS_STR )
    set( CXX_FLAGS ${${PROJECT_NAME}_CXX_FLAGS} )
    if(CXX_FLAGS)
        list( REMOVE_DUPLICATES CXX_FLAGS )
    endif(CXX_FLAGS)
    foreach(FLAG ${CXX_FLAGS})
        set ( PC_CXX_FLAGS_STR "${PC_CXX_FLAGS_STR} ${FLAG}" )
    endforeach(FLAG)

    GET_PROPERTY(GLOBAL_CXX14_FLAG GLOBAL PROPERTY CXX14_FLAG)

    set( LINKER_FLAGS )
    #if ("${CMAKE_EXE_LINKER_FLAGS}" MATCHES "--no-as-needed")
    #    set( LINKER_FLAGS "-Wl,--no-as-needed")
    #endif()

    set( PC_CONTENTS "prefix=${KASPER_INSTALL_DIR}
exec_prefix=${BIN_INSTALL_DIR}
libdir=${LIB_INSTALL_DIR}
includedir=${INCLUDE_INSTALL_DIR}
rpath=${PC_RPATH_STR}

Name: ${PROJECT_NAME}
Description: ${DESCRIPTION}
Version: ${MODULE_VERSION}

Libs: ${LINKER_FLAGS} ${PC_LD_FLAGS_STR} ${PC_LIBRARIES_STR}
Cflags: ${GLOBAL_CXX14_FLAG} ${PC_INCLUDE_DIR_STR} ${PC_CXX_FLAGS_STR}
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
        set(REF_BUILD_DIR ${DOC_INSTALL_DIR}/${PROJECT_NAME}/Reference)
        file(MAKE_DIRECTORY ${REF_BUILD_DIR})
        add_custom_target (reference-${PROJECT_NAME}
            ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Reference/${DOXYGEN_FILE}
            WORKING_DIRECTORY ${REF_BUILD_DIR}
            COMMENT "Generating API documentation with Doxygen for ${PROJECT_NAME}"
            VERBATIM USES_TERMINAL
        )
        add_custom_command(TARGET reference-${PROJECT_NAME} POST_BUILD
            COMMAND ln -sf ${REF_BUILD_DIR}/html/index.html ${DOC_INSTALL_DIR}/${PROJECT_NAME}.html
            DEPENDS ${REF_BUILD_DIR}/html/index.html
            COMMENT "Updating documentation symlinks for ${PROJECT_NAME}"
        )
        add_dependencies(reference reference-${PROJECT_NAME})
        if( ${PROJECT_NAME} STREQUAL Kasper )
            add_dependencies(doc reference-${PROJECT_NAME})
        endif()
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
        set(REF_BUILD_DIR ${DOC_INSTALL_DIR}/${PROJECT_NAME}/Reference)
        file(MAKE_DIRECTORY ${REF_BUILD_DIR})
        add_custom_target (reference-${PROJECT_NAME}
            ${SPHINX_EXECUTABLE} -b html ${CMAKE_CURRENT_BINARY_DIR}/Reference/ ${REF_BUILD_DIR}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Reference/
            COMMENT "Generating API documentation with Sphinx for ${PROJECT_NAME}"
            VERBATIM
        )
        add_dependencies(reference reference-${PROJECT_NAME})
        if( ${PROJECT_NAME} STREQUAL Kasper )
            add_dependencies(doc reference-${PROJECT_NAME})
        endif()
    endif(SPHINX_FOUND)
endmacro()

macro(kasper_add_user_reference_sphinx SPHINX_FILE DOXYGEN_FILE)
    # Builds in the binary tree.
    # This is intended for use by developers only when updating the reference documentation.
    if (SPHINX_FOUND)
        if (DOXYGEN_FOUND)
            if(NOT PACKAGE_VERSION)
                set(PACKAGE_VERSION ${MODULE_VERSION_MAJOR})
            endif()
            #first generate the doxygen C++ API reference
            configure_file (${CMAKE_CURRENT_SOURCE_DIR}/Reference/${DOXYGEN_FILE}.in ${CMAKE_CURRENT_BINARY_DIR}/Reference/${DOXYGEN_FILE} @ONLY)
            set(REF_BUILD_DIR ${DOC_INSTALL_DIR}/${PROJECT_NAME}/UserGuide)
            file(MAKE_DIRECTORY ${REF_BUILD_DIR})
            set(DOXY_REF_BUILD_DIR ${REF_BUILD_DIR}/_static)
            file(MAKE_DIRECTORY ${DOXY_REF_BUILD_DIR})
            add_custom_target (api-reference-${PROJECT_NAME}
                ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Reference/${DOXYGEN_FILE}
                WORKING_DIRECTORY ${DOXY_REF_BUILD_DIR}
                COMMENT "Generating API documentation with Doxygen for ${PROJECT_NAME}"
                VERBATIM
            )
            string(TOLOWER ${PROJECT_NAME} LOWERCASE_PROJECT_NAME)
            ADD_CUSTOM_COMMAND(OUTPUT ${LOWERCASE_PROJECT_NAME}.tag
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/Reference/_static/
                    COMMAND ${CMAKE_COMMAND} -E copy ${DOXY_REF_BUILD_DIR}/${LOWERCASE_PROJECT_NAME}.tag ${CMAKE_CURRENT_BINARY_DIR}/Reference/_static/
                    DEPENDS api-reference-${PROJECT_NAME}
            )
            #careful...this file copy directive is only executed once
            #(it does not update if the .rst source files are changed, only if the build directory is removed!)
            file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Reference/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Reference/)
            add_custom_target (user-reference-${PROJECT_NAME}
                ${SPHINX_EXECUTABLE} -b html ${CMAKE_CURRENT_BINARY_DIR}/Reference/ ${REF_BUILD_DIR}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/Reference/
                COMMENT "Generating user documentation with Sphinx for ${PROJECT_NAME}"
                VERBATIM
                DEPENDS  ${LOWERCASE_PROJECT_NAME}.tag
            )
            add_dependencies(reference user-reference-${PROJECT_NAME})
        endif(DOXYGEN_FOUND)
    endif(SPHINX_FOUND)
endmacro()

macro(kasper_install_doc)
    install(FILES ${ARGN} DESTINATION ${DOC_INSTALL_DIR}/${PROJECT_NAME})
endmacro()
