#ifndef KEMFIELD_CONICSECTION_CL
#define KEMFIELD_CONICSECTION_CL

#include "kEMField_opencl_defines.h"

// Conic section geometry definition (as defined by the streamers in KConicSection.hh):
//
// data[0..2]: P0[0..2]
// data[3..5]: P1[0..2]

//______________________________________________________________________________

void ConicSection_Centroid(CL_TYPE* cen,
			   __global const CL_TYPE* data)
{
    cen[0] = (data[0] + data[3])*.5;
    cen[1] = 0.;
    cen[2] = (data[2] + data[5])*.5;
}

//______________________________________________________________________________

void ConicSection_Normal(CL_TYPE* norm,
			 __global const CL_TYPE* data)
{

}

#endif /* KEMFIELD_CONICSECTION_CL */
