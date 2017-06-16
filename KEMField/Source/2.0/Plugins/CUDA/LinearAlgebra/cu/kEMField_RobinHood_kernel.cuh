#ifndef KEMFIELD_ROBINHOOD_KERNEL_CUH
#define KEMFIELD_ROBINHOOD_KERNEL_CUH

#include "kEMField_cuda_defines.h"

#include "kEMField_LinearAlgebra.cuh"

//______________________________________________________________________________

__forceinline__ __device__
int RH_BoundaryRatioExceeded( int iElement,
        const int* boundaryInfo,
        const int* counter )
{
    int return_val = 0;

    //#ifdef NEUMANNBOUNDARY
    //  int iBoundary = BI_GetBoundaryForElement(iElement,boundaryInfo);
    //
    // is the element a dielectric?
    //  if (BI_GetBoundaryType(iBoundary,boundaryInfo) == NEUMANNBOUNDARY)
    //  {
    // yes.  Is the ratio of times its boundary has been called to the total
    // number of calls more than the ratio of dielectric elements to total
    // elements?
    //
    //    if (counter[BI_GetNumBoundaries(boundaryInfo)]==0)
    //      return return_val;
    //
    //    CU_TYPE ratio_called = 0.;
    //    CU_TYPE ratio_geometric = 0.;
    //
    //    ratio_called = (convert_float(counter[iBoundary])/
    //		    convert_float(counter[BI_GetNumBoundaries(boundaryInfo)]));
    //
    //    ratio_geometric =(convert_float(BI_GetBoundarySize(iBoundary,boundaryInfo))/
    //		      convert_float(BI_GetNumElements(boundaryInfo)));
    //
    //    // this all must be negated if the residual is being checked!
    //    if (ratio_called>ratio_geometric && counter[BI_GetNumBoundaries(boundaryInfo)]%counter[BI_GetNumBoundaries(boundaryInfo)+1]!=0)
    //      return_val = 1;
    //  }
    //#endif

    return return_val;
}

//______________________________________________________________________________

__global__
void InitializeVectorApproximationKernel( const int* boundaryInfo,
        const CU_TYPE* boundaryData,
        const short* shapeInfo,
        const CU_TYPE* shapeData,
        const CU_TYPE* q,
        CU_TYPE* b_diff )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j;
    CU_TYPE value = 0.;

    for( j=0;j<LA_GetVectorSize(boundaryInfo);j++ )
        value += LA_GetMatrixElement( i, j, boundaryInfo, boundaryData, shapeInfo, shapeData) * q[j];

    b_diff[i] = value;

    return;
}

//______________________________________________________________________________

__global__
void FindResidualKernel( CU_TYPE* b_diff,
        const int* boundaryInfo,
        const CU_TYPE* boundaryData,
        const CU_TYPE* b_iterative,
        const int* counter )
{
    int iElement =  blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate deviations
    CU_TYPE U_Target = LA_GetVectorElement(iElement,boundaryInfo,boundaryData);

    if( b_iterative[iElement]<1.e10 ) {
        // if( RH_BoundaryRatioExceeded(iElement,boundaryInfo,counter) )
        //   b_diff[iElement] = 0.;
        // else
        b_diff[iElement] = U_Target - b_iterative[iElement];
    }
    else
        b_diff[iElement] = 0.;
}

//______________________________________________________________________________

__global__
void FindResidualNormKernel( const int* boundaryInfo,
        CU_TYPE* b_diff,
        CU_TYPE* residual )
{
    residual[0] = 0.;
    int i;

    for( i=0; i<LA_GetVectorSize(boundaryInfo); i++ )
        if( FABS(b_diff[i]) > residual[0] )
            residual[0] = FABS(b_diff[i]);
}

//______________________________________________________________________________

__global__
void CompleteResidualNormalizationKernel( const int* boundaryInfo,
        const CU_TYPE* boundaryData,
        CU_TYPE* residual )
{
    int iBoundary;
    CU_TYPE maxPotential = LA_GetMaximumVectorElement( boundaryInfo,boundaryData );
    residual[0] = residual[0]/maxPotential;
}

//______________________________________________________________________________

__global__
void IdentifyLargestResidualElementKernel( CU_TYPE* residual,
        int* partialMaxResidualIndex )
{
    int i =  blockIdx.x * blockDim.x + threadIdx.x;

    int local_i = threadIdx.x;
    int nWorkgroup = blockDim.x;
    int groupID = blockIdx.x;

    extern __shared__  int partialMaxResidualIndex_1Warp[];

    // Reduce
    int tmp = nWorkgroup/2;
    int tmp_last = nWorkgroup;
    int jj = 0;

    partialMaxResidualIndex_1Warp[local_i] = i;

    while( tmp>0 ) {
        __syncthreads();

        if( local_i<tmp ) {
            if( FABS(residual[partialMaxResidualIndex_1Warp[local_i]]) < FABS(residual[partialMaxResidualIndex_1Warp[local_i+tmp]]) ) {
                partialMaxResidualIndex_1Warp[local_i] = partialMaxResidualIndex_1Warp[local_i+tmp];
            }
        }

        if( 2*tmp != tmp_last ) {
            if( local_i==0 ) {
                for (jj=2*tmp;jj<tmp_last;jj++) {
                    if( FABS(residual[partialMaxResidualIndex_1Warp[local_i]]) < FABS(residual[partialMaxResidualIndex_1Warp[jj]]) ) {
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

__global__
void CompleteLargestResidualIdentificationKernel( CU_TYPE* residual,
        const int* boundaryInfo,
        int* partialMaxResidualIndex,
        int* maxResidualIndex,
        const int* nWarps )
{
    // Completes the reduction performed in IdentifyLargestResidualElement()

    CU_TYPE max = -9999;
    maxResidualIndex[0] = 0;
    int maxElement;

    int i;

    for( i=0;i<nWarps[0];i++ ) {
        if( FABS(residual[partialMaxResidualIndex[i]])>max ) {
            maxElement = partialMaxResidualIndex[i];
            max = FABS(residual[partialMaxResidualIndex[i]]);
        }
    }

    maxResidualIndex[0]  = maxElement;
    residual[0] = residual[maxElement];

    return;
}

//______________________________________________________________________________

__global__
void ComputeCorrectionKernel( const short* shapeInfo,
        const CU_TYPE* shapeData,
        const int* boundaryInfo,
        const CU_TYPE* boundaryData,
        CU_TYPE* q,
        const CU_TYPE* b_iterative,
        CU_TYPE* qCorr,
        int* maxResidualIndex,
        int* counter )
{
    int i = maxResidualIndex[0];

    CU_TYPE U_Target = LA_GetVectorElement(i,boundaryInfo,boundaryData);

    CU_TYPE Iii = LA_GetMatrixElement( i,i,boundaryInfo,boundaryData,shapeInfo,shapeData );
    qCorr[0] = (U_Target - b_iterative[i])/Iii;

    // Update counter
    int iBoundary = BI_GetBoundaryForElement(i,boundaryInfo);
    counter[iBoundary] = counter[iBoundary]+1;
    counter[BI_GetNumBoundaries(boundaryInfo)] = counter[BI_GetNumBoundaries(boundaryInfo)]+1;

    return;
}

//______________________________________________________________________________

__global__
void UpdateSolutionApproximationKernel( CU_TYPE* q,
        CU_TYPE* qCorr,
        int* maxResidualIndex )
{
    q[maxResidualIndex[0]] += qCorr[0];
    return;
}

//______________________________________________________________________________

__global__
void UpdateVectorApproximationKernel( const short* shapeInfo,
        const CU_TYPE* shapeData,
        const int* boundaryInfo,
        const CU_TYPE* boundaryData,
        CU_TYPE* b_iterative,
        CU_TYPE* qCorr,
        int* maxResidualIndex )
{
    //   Updates the potential due to change in charge

    int index =  blockIdx.x * blockDim.x + threadIdx.x;

    if (b_iterative[index]>1.e10)
        return;

    int ik = maxResidualIndex[0];

    CU_TYPE Iq_ik = LA_GetMatrixElement(index,ik,boundaryInfo,boundaryData,shapeInfo,shapeData);
    CU_TYPE value = Iq_ik * qCorr[0] + b_iterative[index];
    b_iterative[index] = value;
}

#endif /* KEMFIELD_ROBINHOOD_KERNEL_CUH */
