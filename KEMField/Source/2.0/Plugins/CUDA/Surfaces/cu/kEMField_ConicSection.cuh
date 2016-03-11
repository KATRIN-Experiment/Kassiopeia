#ifndef KEMFIELD_CONICSECTION_CUH
#define KEMFIELD_CONICSECTION_CUH

#include "kEMField_cuda_defines.h"

// Conic section geometry definition (as defined by the streamers in KConicSection.hh):
//
// data[0..2]: P0[0..2]
// data[3..5]: P1[0..2]

//______________________________________________________________________________

__forceinline__ __device__ void ConicSection_Centroid( CU_TYPE* cen, const CU_TYPE* data )
{
    cen[0] = (data[0] + data[3])*.5;
    cen[1] = 0.;
    cen[2] = (data[2] + data[5])*.5;
}

//______________________________________________________________________________

__forceinline__ __device__ void ConicSection_Normal( CU_TYPE* norm, const CU_TYPE* data )
{

}

#endif /* KEMFIELD_CONICSECTION_CUH */
