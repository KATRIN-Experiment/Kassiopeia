#ifndef KEMFIELD_ROBINHOOD_KERNEL_CL
#define KEMFIELD_ROBINHOOD_KERNEL_CL

#include "kEMField_opencl_defines.h"

#include "kEMField_LinearAlgebra.cl"

//______________________________________________________________________________

int RH_BoundaryRatioExceeded(int iElement,
                 __global const int* boundaryInfo,
                 __global const int* counter)
{
  int return_val = 0;

#ifdef NEUMANNBOUNDARY
  int iBoundary = BI_GetBoundaryForElement(iElement,boundaryInfo);

//   is the element a dielectric?
  if (BI_GetBoundaryType(iBoundary,boundaryInfo) == NEUMANNBOUNDARY)
  {
//     yes.  Is the ratio of times its boundary has been called to the total
//     number of calls more than the ratio of dielectric elements to total
//     elements?

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
#ifdef NEUMANNBOUNDARY
#if KEMFIELD_OCLNEUMANNCHECKMETHOD==1
     // Decrease checked accuracy of Neumann elements by 1/20 (idea by Ferenc Glueck)
     int iBoundary = BI_GetBoundaryForElement(iElement,boundaryInfo);
     if (BI_GetBoundaryType(iBoundary,boundaryInfo) == NEUMANNBOUNDARY)
       b_diff[iElement] = 0.05 * (U_Target - b_iterative[iElement]);
     else
#endif /* KEMFIELD_OCLNEUMANNCHECKMETHOD==1 */
#if KEMFIELD_OCLNEUMANNCHECKMETHOD==2
     // Counter technique with the function RH_BoundaryRatioExceeded (by T.J. Corona)
     if (RH_BoundaryRatioExceeded(iElement,boundaryInfo,counter))
       b_diff[iElement] = 0.;
     else
#endif /* KEMFIELD_OCLNEUMANNCHECKMETHOD==2 */
#endif
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
      residual[0] = fabs(b_diff[i]);
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

__kernel void IdentifyLargestResidualElement(__global CL_TYPE* residual,
					     __local int* partialMaxResidualIndex_1Warp,
					     __global int* partialMaxResidualIndex)
{
  int i = get_global_id(0);

  int local_i = get_local_id(0);
  int nWorkgroup = get_local_size(0);
  int groupID = get_group_id(0);

  // Reduce
  int tmp = nWorkgroup/2;
  int tmp_last = nWorkgroup;
  int jj = 0;

  partialMaxResidualIndex_1Warp[local_i] = i;

  while (tmp>0)
  {
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_i<tmp)
    {
      if (fabs(residual[partialMaxResidualIndex_1Warp[local_i]]) <
	  fabs(residual[partialMaxResidualIndex_1Warp[local_i+tmp]]))
      {
	partialMaxResidualIndex_1Warp[local_i] = partialMaxResidualIndex_1Warp[local_i+tmp];
      }
    }
    if (2*tmp != tmp_last)
    {
      if (local_i==0)
      {
	for (jj=2*tmp;jj<tmp_last;jj++)
	{
	  if (fabs(residual[partialMaxResidualIndex_1Warp[local_i]]) <
	      fabs(residual[partialMaxResidualIndex_1Warp[jj]]))
	  {
	    partialMaxResidualIndex_1Warp[local_i] = partialMaxResidualIndex_1Warp[jj];
	  }
	}
      }
    }

    tmp_last = tmp;
    tmp/=2;
  }

  if (local_i==0)
    partialMaxResidualIndex[groupID] = partialMaxResidualIndex_1Warp[local_i];
}

//______________________________________________________________________________

__kernel void CompleteLargestResidualIdentification(__global CL_TYPE* residual,
						    __global const int* boundaryInfo,
						    __global int* partialMaxResidualIndex,
						    __global int* maxResidualIndex,
						    __global const int* nWarps)
{
  // Completes the reduction performed in IdentifyLargestResidualElement()

  CL_TYPE max = -9999;
  maxResidualIndex[0] = 0;
  int maxElement;

  int i;

  for (i=0;i<nWarps[0];i++)
  {
    if (fabs(residual[partialMaxResidualIndex[i]])>max)
    {
      maxElement = partialMaxResidualIndex[i];
      max = fabs(residual[partialMaxResidualIndex[i]]);
    }
  }

  maxResidualIndex[0]  = maxElement;
  residual[0] = residual[maxElement];

  return;
}

//______________________________________________________________________________

__kernel void ComputeCorrection(__global const short* shapeInfo,
				__global const CL_TYPE* shapeData,
				__global const int* boundaryInfo,
				__global const CL_TYPE* boundaryData,
				__global CL_TYPE* q,
				__global const CL_TYPE* b_iterative,
				__global CL_TYPE* qCorr,
				__global int* maxResidualIndex,
				__global int* counter)
{
  int i = maxResidualIndex[0];

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
					  __global int* maxResidualIndex)
{
  q[maxResidualIndex[0]] += qCorr[0];
  return;
}

//______________________________________________________________________________

__kernel void UpdateVectorApproximation(__global const short* shapeInfo,
					__global const CL_TYPE* shapeData,
					__global const int* boundaryInfo,
					__global const CL_TYPE* boundaryData,
					__global CL_TYPE* b_iterative,
					__global CL_TYPE* qCorr,
					__global int* maxResidualIndex)
{
  //   Updates the potential due to change in charge

  int index = get_global_id(0);

  if (b_iterative[index]>1.e10)
    return;

  int ik = maxResidualIndex[0];

  CL_TYPE Iq_ik = LA_GetMatrixElement(index,ik,boundaryInfo,boundaryData,shapeInfo,shapeData);
  CL_TYPE value = Iq_ik * qCorr[0] + b_iterative[index];
  b_iterative[index] = value;
}

#endif /* KEMFIELD_ROBINHOOD_KERNEL_CL */
