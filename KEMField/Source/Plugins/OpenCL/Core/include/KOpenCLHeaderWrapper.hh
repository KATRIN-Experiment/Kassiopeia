#ifndef KOPENCLHEADERWRAPPER_DEF
#define KOPENCLHEADERWRAPPER_DEF

#ifdef __clang__
#pragma clang system_header
#endif
#ifdef __GNUG__
#pragma GCC system_header
#endif

// use OpenCL 1.2 definitions - higher version are not working correctly
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 120

#if defined __APPLE__
#include <OpenCL/cl2.hpp>
#else
#include <CL/cl2.hpp>
#endif

#include "kEMField_opencl_defines.h"

#endif /* KOPENCLHEADERWRAPPER_DEF */
