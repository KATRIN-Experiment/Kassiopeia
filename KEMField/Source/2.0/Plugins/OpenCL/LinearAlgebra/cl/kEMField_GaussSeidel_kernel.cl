#ifndef KEMFIELD_GAUSSSEIDEL_KERNEL_CL
#define KEMFIELD_GAUSSSEIDEL_KERNEL_CL

#include "kEMField_defines.h"

#include "kEMField_LinearAlgebra.cl"

//______________________________________________________________________________

int GS_BoundaryRatioExceeded(int iElement,
			     __global const int* boundaryInfo,
			     __global const int* counter)
{
  int return_val = 0;

#ifdef NEUMANNBOUNDARY
  int iBoundary = BI_GetBoundaryForElement(iElement,boundaryInfo);

  // is the element a dielectric?
  if (BI_GetBoundaryType(iBoundary,boundaryInfo) == NEUMANNBOUNDARY)
  {
    // yes.  Is the ratio of times its boundary has been called to the total
    // number of calls more than the ratio of dielectric elements to total
    // elements?

    if (counter[BI_GetNumBoundaries(boundaryInfo)]==0)
      return return_val;

    CL_TYPE ratio_called = 0.;
    CL_TYPE ratio_geometric = 0.;

    ratio_called = (convert_float(counter[iBoundary])/
		    convert_float(counter[BI_GetNumBoundaries(boundaryInfo)]));

    ratio_geometric =(convert_float(BI_GetBoundarySize(iBoundary,boundaryInfo))/
		      convert_float(BI_GetNumElements(boundaryInfo)));

    // this all must be negated if the residual is being checked!
    if (ratio_called>ratio_geometric && counter[BI_GetNumBoundaries(boundaryInfo)]%counter[BI_GetNumBoundaries(boundaryInfo)+1]!=0)
      return_val = 1;
  }
#endif

  return return_val;
}

//______________________________________________________________________________

__kernel void InitializeVectorApproximation(__global const int* boundaryInfo,
					    __global const CL_TYPE* boundaryData,
					    __global const short* shapeInfo,
					    __global const CL_TYPE* shapeData,
					    __global const CL_TYPE* q,
					    __global CL_TYPE* b_diff)
{
  int i = get_global_id(0);
  int j;
  CL_TYPE value = 0.;
  for (j=0;j<LA_GetVectorSize(boundaryInfo);j++)
    value += LA_GetMatrixElement(i,
				 j,
				 boundaryInfo,
				 boundaryData,
				 shapeInfo,
				 shapeData)*q[j];

  b_diff[i] = value;

  return;
}

//______________________________________________________________________________

__kernel void FindResidual(__global CL_TYPE* b_diff,
			   __global const int* boundaryInfo,
			   __global const CL_TYPE* boundaryData,
			   __global const CL_TYPE* b_iterative,
			   __global const int* counter)
{
  int iElement = get_global_id(0);

  // Calculate deviations
  CL_TYPE U_Target = LA_GetVectorElement(iElement,boundaryInfo,boundaryData);

  if (b_iterative[iElement]<1.e10)
  {
    // if (GS_BoundaryRatioExceeded(iElement,boundaryInfo,counter))
    //   b_diff[iElement] = 0.;
    // else
      b_diff[iElement] = U_Target - b_iterative[iElement];
  }
  else
    b_diff[iElement] = 0.;
}

//______________________________________________________________________________

__kernel void FindResidualNorm(__global const int* boundaryInfo,
			       __global CL_TYPE* b_diff,
			       __global CL_TYPE* residual)
{
  residual[0] = 0.;
  int i;
  for (i=0;i<LA_GetVectorSize(boundaryInfo);i++)
    if (fabs(b_diff[i]) > residual[0])
      residual[0] = b_diff[i];
}

//______________________________________________________________________________

__kernel void CompleteResidualNormalization(__global const int* boundaryInfo,
					    __global const CL_TYPE* boundaryData,
					    __global CL_TYPE* residual)
{
  int iBoundary;
  CL_TYPE maxPotential = LA_GetMaximumVectorElement(boundaryInfo,boundaryData);
  residual[0] = residual[0]/maxPotential;
}

//______________________________________________________________________________

__kernel void IncrementIndex(__global const int* boundaryInfo,
			     __global int* index)
{
  index[0] = index[0] + 1;
  if (index[0] == LA_GetVectorSize(boundaryInfo))
    index[0] = 0;
}

//______________________________________________________________________________

__kernel void ComputeCorrection(__global const short* shapeInfo,
				__global const CL_TYPE* shapeData,
				__global const int* boundaryInfo,
				__global const CL_TYPE* boundaryData,
				__global CL_TYPE* q,
				__global const CL_TYPE* b_iterative,
				__global CL_TYPE* qCorr,
				__global int* index,
				__global int* counter)
{
  int i = index[0];

  CL_TYPE U_Target = LA_GetVectorElement(i,boundaryInfo,boundaryData);

  CL_TYPE Iii = LA_GetMatrixElement(i,i,boundaryInfo,boundaryData,shapeInfo,shapeData);
  qCorr[0] = (U_Target - b_iterative[i])/Iii;

  // Update counter
  int iBoundary = BI_GetBoundaryForElement(i,boundaryInfo);
  counter[iBoundary] = counter[iBoundary]+1;
  counter[BI_GetNumBoundaries(boundaryInfo)] = counter[BI_GetNumBoundaries(boundaryInfo)]+1;

  return;
}

//______________________________________________________________________________

__kernel void UpdateSolutionApproximation(__global CL_TYPE* q,
					  __global CL_TYPE* qCorr,
					  __global int* index)
{
  q[index[0]] += qCorr[0];
  return;
}

//______________________________________________________________________________

__kernel void UpdateVectorApproximation(__global const short* shapeInfo,
					__global const CL_TYPE* shapeData,
					__global const int* boundaryInfo,
					__global const CL_TYPE* boundaryData,
					__global CL_TYPE* b_iterative,
					__global CL_TYPE* qCorr,
					__global int* index)
{
  //   Updates the potential due to change in charge

  int localIndex = get_global_id(0);

  if (b_iterative[localIndex]>1.e10)
    return;

  int ik = index[0];

  CL_TYPE Iq_ik = LA_GetMatrixElement(localIndex,ik,boundaryInfo,boundaryData,shapeInfo,shapeData);
  CL_TYPE value = Iq_ik * qCorr[0] + b_iterative[localIndex];
  b_iterative[localIndex] = value;
}

#endif /* KEMFIELD_GAUSSSEIDEL_KERNEL_CL */
