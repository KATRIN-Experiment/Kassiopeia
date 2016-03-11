/*
 * kEMField_cuda_defines.h
 *
 *  Created on: 22.06.2015
 *      Author: Daniel Hilk
 */

#ifndef KEMFIELD_CUDA_DEFINES_H_
#define KEMFIELD_CUDA_DEFINES_H_

// enabled in default via math.h : #define M_PI 3.141592653589793238462643
#define M_EPS0 8.85418782e-12
#define M_PI_OVER_2 1.570796326794896619231321

#ifdef KEMFIELD_USE_DOUBLE_PRECISION
#define CU_TYPE double
#define CU_TYPE4 double4
#define SQRT(x) sqrt(x)
#define LOG(x) log(x)
#define ATAN(x) atan(x)
#define FABS(x) fabs(x)
#else
#define CU_TYPE float
#define CU_TYPE4 float4
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
//#define CONICSECTION 3
#define DIRICHLETBOUNDARY 0
#define NEUMANNBOUNDARY 1

#endif /* KEMFIELD_CUDA_DEFINES_H_ */
