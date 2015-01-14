#ifndef KEMFIELD_LINESEGMENT_CL
#define KEMFIELD_LINESEGMENT_CL

#include "kEMField_defines.h"

// Wire geometry definition (as defined by the streamers in KLineSegment.hh):
//
// data[0..2]: P0[0..2]
// data[3..5]: P1[0..2]
// data[6]:    diameter

//______________________________________________________________________________

void Line_Centroid(CL_TYPE* cen,
		   __global const CL_TYPE* data)
{
  cen[0] = (data[0] + data[3])*.5;
  cen[1] = (data[1] + data[4])*.5;
  cen[2] = (data[2] + data[5])*.5;
}

//______________________________________________________________________________

void Line_Normal(CL_TYPE* norm,
		 __global const CL_TYPE* data)
{

}

#endif /* KEMFIELD_LINESEGMENT_CL */
