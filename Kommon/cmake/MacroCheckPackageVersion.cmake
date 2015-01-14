##############################################################################
# macro for checking a package version
#
# this macro should be called from your PKGVersion.cmake or from a
#   FindPKG.cmake module with the following arguments:
#       _pkgname    : The package name
#       _iversion   : The installed version of the package
#
#
# the following conventions are used:
#
#   if FIND_PACKAGE is called with EXACT argument than the version has to
#   match EXACTLY, i.e.:
#       1.5 == 1.5
#       1.5 == 1.5.0
#       1.5 == 1.5.0.0
#       1.5.2 == 1.5.2.0
#       1.5.2.1 == 1.5.2.1
#       1.5.2 != 1.5.2.1
#       1.5 != 1.5.0.1
#
#
#   otherwise a MINIMUM_REQUIRED version is checked for, i.e. the same
#   behavior as with the cmake variable CMAKE_MINIMUM_REQUIRED, e.g.:
#       searching: 1.2     --> installed: 1.5.2.2 --> compatible
#       searching: 1.5     --> installed: 1.5.2.2 --> compatible
#       searching: 1.5.2.1 --> installed: 1.5.2.2 --> compatible
#       searching: 1.5.2.3 --> installed: 1.5.2.2 --> unsuitable
#       searching: 1.7     --> installed: 1.5.2.2 --> unsuitable
#
#
# following variables are returned (internally to cmake):
#   PACKAGE_VERSION_EXACT       : set to TRUE if exact version was found
#   PACKAGE_VERSION_COMPATIBLE  : set to TRUE if version is compatible
#   PACKAGE_VERSION_UNSUITABLE  : set to TRUE if version found is unsuitable
#
#
# @author Jan Engels, Desy IT
##############################################################################

# these variables are evaluated internally by the cmake command FIND_PACKAGE to mark this
# package as suitable or not depending on the required version
SET( PACKAGE_VERSION_EXACT FALSE )
SET( PACKAGE_VERSION_COMPATIBLE TRUE )
SET( PACKAGE_VERSION_UNSUITABLE FALSE )


# cmake internal variable PACKAGE_FIND_NAME is not defined on FindPKG.cmake
# modules, therefore it is passed as an argument to the macro
# _iversion is the installed version of the package
# _sversion is the version searched by the user with FIND_PACKAGE
#MACRO( CHECK_PACKAGE_VERSION _pkgname _iversion )
MACRO( CHECK_PACKAGE_VERSION _pkgname ) # left with one argument only for backwards compatibility

    IF( NOT "${ARGV1}" STREQUAL "" )
        SET( _iversion ${ARGV1} )
    ELSE()
        SET( _iversion ${${_pkgname}_VERSION_MAJOR}.${${_pkgname}_VERSION_MINOR}.${${_pkgname}_VERSION_PATCH}.${${_pkgname}_VERSION_TWEAK} )
    ENDIF()

    #SET( _sversion_major ${${_pkgname}_FIND_VERSION_MAJOR} )
    #SET( _sversion_minor ${${_pkgname}_FIND_VERSION_MINOR} )

    SET( _sversion ${${_pkgname}_FIND_VERSION} )

    IF( NOT ${_pkgname}_FIND_QUIETLY )
        MESSAGE( STATUS "Check for ${_pkgname} (${_iversion})" )
    ENDIF()

    # only do work if FIND_PACKAGE called with a version argument
    IF( _sversion )

        #IF( NOT ${_pkgname}_FIND_QUIETLY )
        #    MESSAGE( STATUS "Check for ${_pkgname}: looking for version ${_sversion}" )
        #ENDIF()

        IF( ${_iversion} VERSION_EQUAL ${_sversion} ) # if version matches EXACTLY
            #IF( NOT ${_pkgname}_FIND_QUIETLY )
            #    MESSAGE( STATUS "Check for ${_pkgname}: exact version found: ${_iversion}" )
            #ENDIF()
            SET( PACKAGE_VERSION_EXACT TRUE )
        ELSE() # if version does not match EXACTLY, check if it is compatible/suitable

            # installed version must be greater or equal than version searched by the user, i.e.
            # like with the CMAKE_MINIMUM_REQUIRED commando
            # if user asks for version 1.2.5 then any version >= 1.2.5 is suitable/compatible
            IF( NOT ${_sversion} VERSION_LESS ${_iversion} )
                SET( PACKAGE_VERSION_UNSUITABLE TRUE )
            ENDIF()
            # -------------------------------------------------------------------------------------

            IF( ${_pkgname}_FIND_VERSION_EXACT ) # if exact version was required search must fail!!
                #IF( NOT ${_pkgname}_FIND_QUIETLY )
                #    MESSAGE( "Check for ${_pkgname}: could not find exact version" )
                #ENDIF()
                SET( PACKAGE_VERSION_UNSUITABLE TRUE )
            ENDIF()

        ENDIF()

        IF( PACKAGE_VERSION_UNSUITABLE )
            SET( PACKAGE_VERSION_COMPATIBLE FALSE )
        ENDIF()

    ENDIF( _sversion )

ENDMACRO( CHECK_PACKAGE_VERSION )

