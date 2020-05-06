#ifndef KOPENCLHEADERWRAPPER_DEF
#define KOPENCLHEADERWRAPPER_DEF

#ifdef __clang__
#pragma clang system_header
#endif
#ifdef __GNUG__
#pragma GCC system_header
#endif

#if defined __APPLE__
#include <OpenCL/cl.hpp>
#else
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.hpp>
#endif

#endif /* KOPENCLHEADERWRAPPER_DEF */
