#ifndef KEMFIELD_LINESEGMENT_CUH
#define KEMFIELD_LINESEGMENT_CUH

#include "kEMField_cuda_defines.h"

// Wire geometry definition (as defined by the streamers in KLineSegment.hh):
//
// data[0..2]: P0[0..2]
// data[3..5]: P1[0..2]
// data[6]:    diameter

//______________________________________________________________________________

__forceinline__ __device__ void Line_Centroid( CU_TYPE* cen, const CU_TYPE* data )
{
    cen[0] = (data[0] + data[3])*.5;
    cen[1] = (data[1] + data[4])*.5;
    cen[2] = (data[2] + data[5])*.5;
}

//______________________________________________________________________________

__forceinline__ __device__ void Line_Normal( CU_TYPE* norm, const CU_TYPE* data )
{

}

#endif /* KEMFIELD_LINESEGMENT_CUH */
