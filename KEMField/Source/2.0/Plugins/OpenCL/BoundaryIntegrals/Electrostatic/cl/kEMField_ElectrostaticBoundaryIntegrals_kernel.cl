#ifndef KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_KERNEL_CL
#define KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_KERNEL_CL

#include "kEMField_opencl_defines.h"

#include "kEMField_ElectrostaticBoundaryIntegrals.cl"

//______________________________________________________________________________

__kernel void Potential(__global const CL_TYPE *P,
			__global const short *shapeType,
			__global const CL_TYPE *data,
			__global CL_TYPE* phi)
{
  CL_TYPE p[3] = {P[0],P[1],P[2]};
  phi[0] = EBI_Potential(p,shapeType,data);
}

//______________________________________________________________________________

__kernel void ElectricField(__global const CL_TYPE *P,
			    __global const short *shapeType,
			    __global const CL_TYPE *data,
			    __global CL_TYPE4 *eField)
{
  CL_TYPE p[3] = {P[0],P[1],P[2]};
  eField[0] = EBI_EField(p,shapeType,data);
}

//______________________________________________________________________________

__kernel void ElectricFieldAndPotential(__global const CL_TYPE *P,
			    __global const short *shapeType,
			    __global const CL_TYPE *data,
			    __global CL_TYPE4 *eFieldAndPhi)
{
  CL_TYPE p[3] = {P[0],P[1],P[2]};
  eFieldAndPhi[0] = EBI_EFieldAndPotential(p,shapeType,data);
}

#endif /* KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_KERNEL_CL */
