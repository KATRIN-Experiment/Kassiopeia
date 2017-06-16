/*
 * kEMField_cuda_defines.h
 *
 *  Created on: 22.06.2015
 *      Author: Daniel Hilk
 */

#ifndef KEMFIELD_CUDA_DEFINES_H_
#define KEMFIELD_CUDA_DEFINES_H_

#define M_EPS0 8.854187817E-12
#define M_PI_OVER_2 1.57079632679489661923
#define M_ONEOVER_4PI_EPS0 8987551787.9979107161559640186992

#define POW2(x) ((x)*(x))
#define POW3(x) ((x)*(x)*(x))

#ifdef KEMFIELD_USE_DOUBLE_PRECISION
#define CU_TYPE double
#define CU_TYPE4 double4
#define MAKECU4 make_double4
#define SQRT(x) sqrt(x)
#define LOG(x) log(x)
#define ATAN(x) atan(x)
#define FABS(x) fabs(x)
#else
#define CU_TYPE float
#define CU_TYPE4 float4
#define MAKECU4 make_float4
#define SQRT(x) sqrtf(x)
#define LOG(x) logf(x)
#define ATAN(x) atanf(x)
#define FABS(x) fabsf(x)
#endif

#define SHAPESIZE 11
#define BOUNDARYSIZE 1
#define BASISSIZE 1

#define TRIANGLE 0
#define RECTANGLE 1
#define LINESEGMENT 2
// conic sections have been deactivated since related
// macro expansions lead to long compilation times
// #define CONICSECTION 3

#define DIRICHLETBOUNDARY 0
#define NEUMANNBOUNDARY 1

#endif /* KEMFIELD_CUDA_DEFINES_H_ */
