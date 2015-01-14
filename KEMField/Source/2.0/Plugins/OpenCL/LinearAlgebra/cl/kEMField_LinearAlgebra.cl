#ifndef KEMFIELD_LINEARALGEBRA_CL
#define KEMFIELD_LINEARALGEBRA_CL

#include "kEMField_defines.h"

#include KEMFIELD_INTEGRATORFILE_CL

//______________________________________________________________________________

CL_TYPE LA_GetMatrixElement(int i, // target
			    int j, // source
			    __global const int* boundaryInfo,
			    __global const CL_TYPE* boundaryData,
			    __global const short* shapeInfo,
			    __global const CL_TYPE* shapeData)
{
  int iBoundary = BI_GetBoundaryForElement(i,boundaryInfo);

  return BI_BoundaryIntegral(iBoundary,
			     boundaryInfo,
			     boundaryData,
			     &shapeInfo[i],
			     &shapeInfo[j],
			     &shapeData[i*SHAPESIZE],
			     &shapeData[j*SHAPESIZE]);
}

//______________________________________________________________________________

int LA_GetVectorSize(__global const int* boundaryInfo)
{
  return BI_GetNumElements(boundaryInfo);
}

//______________________________________________________________________________

CL_TYPE LA_GetVectorElement(int i,
			    __global const int* boundaryInfo,
			    __global const CL_TYPE* boundaryData)
{
  int iBoundary = BI_GetBoundaryForElement(i,boundaryInfo);

  return BI_GetBoundaryValue(iBoundary,
			     boundaryInfo,
			     boundaryData);
}

//______________________________________________________________________________

CL_TYPE LA_GetMaximumVectorElement(__global const int* boundaryInfo,
				   __global const CL_TYPE* boundaryData)
{
  CL_TYPE max = -1.e30;

  int iBoundary;
  for(iBoundary = 0; iBoundary < BI_GetNumBoundaries(boundaryInfo); iBoundary++)
  {
    if (max<fabs(BI_GetBoundaryValue(iBoundary,boundaryInfo,boundaryData))) 
      max = fabs(BI_GetBoundaryValue(iBoundary,boundaryInfo,boundaryData));
  }

  if (max < 1.e-10) max = 1.;

  return max;
}

//______________________________________________________________________________

CL_TYPE LA_GetMaximumSolutionVectorElement(__global const int* boundaryInfo,
					   __global const CL_TYPE* basisData)
{
  CL_TYPE max = -1.e30;

  int i;
  for(i = 0; i < LA_GetVectorSize(boundaryInfo); i++)
  {
    if (max<fabs(basisData[i]))
      max = fabs(basisData[i]);
  }

  return max;
}

#endif /* KEMFIELD_LINEARALGEBRA_CL */
