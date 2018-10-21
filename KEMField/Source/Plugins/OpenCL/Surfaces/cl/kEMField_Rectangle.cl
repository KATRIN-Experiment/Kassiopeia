#ifndef KEMFIELD_RECTANGLE_CL
#define KEMFIELD_RECTANGLE_CL

#include "kEMField_opencl_defines.h"

// Rectangle geometry definition (as defined by the streamers in KRectangle.hh):
//
// data[0]:     A
// data[1]:     B
// data[2..4]:  P0[0..2]
// data[5..7]:  N1[0..2]
// data[8..10]: N2[0..2]

//______________________________________________________________________________

void Rect_Centroid(CL_TYPE* cen,
		   __global const CL_TYPE* data)
{
  cen[0] = data[2] + data[0]*data[5]*.5 + data[1]*data[8]*.5;
  cen[1] = data[3] + data[0]*data[6]*.5 + data[1]*data[9]*.5;
  cen[2] = data[4] + data[0]*data[7]*.5 + data[1]*data[10]*.5;
}

//______________________________________________________________________________

void Rect_Normal(CL_TYPE* norm,
	       __global const CL_TYPE* data)
{
  norm[0] = data[6]*data[10] - data[7]*data[9];
  norm[1] = data[7]*data[8]  - data[5]*data[10];
  norm[2] = data[5]*data[9]  - data[6]*data[8];
}

#endif /* KEMFIELD_RECTANGLE_CL */
