#ifndef KEMFIELD_LINEARALGEBRA_KERNEL_CL
#define KEMFIELD_LINEARALGEBRA_KERNEL_CL

#include "kEMField_LinearAlgebra.cl"

//______________________________________________________________________________

__kernel void GetMatrixElement(__global int* ij,
			       __global const int* boundaryInfo,
			       __global const CL_TYPE* boundaryData,
			       __global const short* shapeInfo,
			       __global const CL_TYPE* shapeData,
			       __global CL_TYPE* value)
{
  value[0] = LA_GetMatrixElement(ij[0],
  				 ij[1],
  				 boundaryInfo,
  				 boundaryData,
  				 shapeInfo,
  				 shapeData);
  return;
}

//______________________________________________________________________________

__kernel void GetVectorElement(__global int* i,
			       __global const int* boundaryInfo,
			       __global const CL_TYPE* boundaryData,
			       __global CL_TYPE* value)
{
  value[0] = LA_GetVectorElement(i[0],
				 boundaryInfo,
				 boundaryData);
  return;
}

//______________________________________________________________________________

__kernel void GetSolutionVectorElement(__global int* i,
				       __global CL_TYPE* q,
				       __global CL_TYPE* value)
{
  value[0] = q[i[0]];
  return;
}

//______________________________________________________________________________

__kernel void GetMaximumVectorElement(__global const int* boundaryInfo,
				      __global const CL_TYPE* boundaryData,
				      __global CL_TYPE* value)
{
  value[0] = LA_GetMaximumVectorElement(boundaryInfo,boundaryData);
  return;
}

//______________________________________________________________________________

__kernel void GetMaximumSolutionVectorElement(__global const int* boundaryInfo,
					      __global const CL_TYPE* basisData,
					      __global CL_TYPE* value)
{
  value[0] = LA_GetMaximumSolutionVectorElement(boundaryInfo,basisData);
  return;
}

#endif /* KEMFIELD_LINEARALGEBRA_KERNEL_CL */
