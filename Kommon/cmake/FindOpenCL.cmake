# - Try to find OpenCL
# This module tries to find an OpenCL implementation on your system. It supports
# AMD / ATI, Apple and NVIDIA implementations, but should work, too.
#
# To set manually the paths, define these environment variables:
# OpenCL_INCPATH    - Include path (e.g. OpenCL_INCPATH=/opt/cuda/4.0/cuda/include)
# OpenCL_LIBPATH    - Library path (e.h. OpenCL_LIBPATH=/usr/lib64/nvidia)
#
# Once done this will define
#  OPENCL_FOUND        - system has an OpenCL library
#  OPENCL_INCLUDE_DIRS  - the OpenCL include directory
#  OPENCL_LIBRARIES    - link these to use OpenCL
#  OPENCL_LIB_FLAG compiler flag naming library
#
# WIN32 should work, but is untested

FIND_PACKAGE(PackageHandleStandardArgs)

SET (OPENCL_VERSION_STRING "0.1.0")
SET (OPENCL_VERSION_MAJOR 0)
SET (OPENCL_VERSION_MINOR 1)
SET (OPENCL_VERSION_PATCH 0)

IF (APPLE)

	FIND_LIBRARY(OPENCL_LIBRARIES OpenCL DOC "OpenCL lib for OSX")
	FIND_PATH(OPENCL_INCLUDE_DIRS OpenCL/cl.h DOC "Include for OpenCL on OSX")
	FIND_PATH(_OPENCL_CPP_INCLUDE_DIRS OpenCL/cl.hpp DOC "Include for OpenCL CPP bindings on OSX")

ELSE (APPLE)

	#Unix style platforms, no windows support

    #set the library flag to the compiler to the default value
    set(OPENCL_LIBRARY_NAME "OpenCL" CACHE STRING "Name of the OpenCL library.")
    #leave the option to set the name of the library directly
    #because some systems use a different name, such as 'intelocl'
    mark_as_advanced(OPENCL_LIBRARY_NAME)

    #force cmake to search for the library every time
    set (OPENCL_LIBRARIES "OPENCL_LIBRARY-NOTFOUND")

	FIND_LIBRARY(OPENCL_LIBRARIES ${OPENCL_LIBRARY_NAME}
		PATHS ENV LD_LIBRARY_PATH ENV OpenCL_LIBPATH ENV OPENCL_LIB_PATH
	)

    message(STATUS "Looking for an OpenCL library named: ${OPENCL_LIBRARY_NAME}")

	GET_FILENAME_COMPONENT(OPENCL_LIB_DIR ${OPENCL_LIBRARIES} PATH)
	GET_FILENAME_COMPONENT(_OPENCL_INC_CAND ${OPENCL_LIB_DIR}/../../include ABSOLUTE)

    message(STATUS "Expecting OpenCL library in: ${OPENCL_LIB_DIR}")
    message(STATUS "OpenCL libary found is: ${OPENCL_LIBRARIES}")

    #if the system has an environmental variable called OPENCL_INCLUDE
    #which is the compiler flag for the include files, we strip the leading -I
    #to get the include path
    string(REPLACE "-I" "" OPENCL_INCLUDE_FLAG_DIR "$ENV{OPENCL_INCLUDE}")
    message(STATUS "Found compiler flag to OpenCL include directory: " ${OPENCL_INCLUDE_FLAG_DIR})

	# The AMD SDK currently does not place its headers
	# in /usr/include, therefore also search relative
	# to the library
	FIND_PATH( OPENCL_INCLUDE_DIRS CL/cl.h
        PATHS ${_OPENCL_INC_CAND} "/usr/local/cuda/include" "/opt/AMDAPP/include"
        ENV OpenCL_INCPATH ${OPENCL_INCLUDE_FLAG_DIR}
        )
	FIND_PATH( _OPENCL_CPP_INCLUDE_DIRS CL/cl.hpp
        PATHS ${_OPENCL_INC_CAND} "/usr/local/cuda/include" "/opt/AMDAPP/include"
        ENV OpenCL_INCPATH ENV ${OPENCL_INCLUDE_FLAG_DIR}
        )

    message(STATUS "Expecting OpenCL headers in: " ${OPENCL_INCLUDE_DIRS})

    set(OPENCL_LIB_FLAG "-l${OPENCL_LIBRARY_NAME}")

    message(STATUS "opencl library is: ${OPENCL_LIBRARIES}")


ENDIF (APPLE)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(${OPENCL_LIBRARY_NAME} DEFAULT_MSG OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS)

IF(_OPENCL_CPP_INCLUDE_DIRS)
	SET( OPENCL_HAS_CPP_BINDINGS TRUE )
	LIST( APPEND OPENCL_INCLUDE_DIRS ${_OPENCL_CPP_INCLUDE_DIRS} )
	# This is often the same, so clean up
	LIST( REMOVE_DUPLICATES OPENCL_INCLUDE_DIRS )
ENDIF(_OPENCL_CPP_INCLUDE_DIRS)


MARK_AS_ADVANCED(
  OPENCL_INCLUDE_DIRS
)
