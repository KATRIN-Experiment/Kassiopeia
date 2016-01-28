#ifndef KEMFIELD_LINEARALGEBRA_CUH
#define KEMFIELD_LINEARALGEBRA_CUH

#include "kEMField_cuda_defines.h"

// todo: define this dynamically
#include "kEMField_ElectrostaticBoundaryIntegrals.cuh"

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE LA_GetMatrixElement( int i, // target
        int j, // source
        const int* boundaryInfo,
        const CU_TYPE* boundaryData,
        const short* shapeInfo,
        const CU_TYPE* shapeData )
{
    int iBoundary = BI_GetBoundaryForElement(i,boundaryInfo);

    return BI_BoundaryIntegral( iBoundary,
            boundaryInfo,
            boundaryData,
            &shapeInfo[i],
            &shapeInfo[j],
            &shapeData[i*SHAPESIZE],
            &shapeData[j*SHAPESIZE] );
}

//______________________________________________________________________________

 __forceinline__ __device__ int LA_GetVectorSize( const int* boundaryInfo )
{
    return BI_GetNumElements( boundaryInfo );
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE LA_GetVectorElement( int i,
        const int* boundaryInfo,
        const CU_TYPE* boundaryData )
{
    int iBoundary = BI_GetBoundaryForElement( i,boundaryInfo );

    return BI_GetBoundaryValue( iBoundary, boundaryInfo, boundaryData );
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE LA_GetMaximumVectorElement( const int* boundaryInfo,
        const CU_TYPE* boundaryData )
{
    CU_TYPE max = -1.e30;

    int iBoundary;
    for( iBoundary = 0; iBoundary < BI_GetNumBoundaries(boundaryInfo); iBoundary++ ) {
        if (max<FABS(BI_GetBoundaryValue(iBoundary,boundaryInfo,boundaryData)))
            max = FABS(BI_GetBoundaryValue(iBoundary,boundaryInfo,boundaryData));
    }

    if (max < 1.e-10) max = 1.;

    return max;
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE LA_GetMaximumSolutionVectorElement( const int* boundaryInfo,
        const CU_TYPE* basisData)
{
    CU_TYPE max = -1.e30;

    int i;
    for(i = 0; i < LA_GetVectorSize(boundaryInfo); i++) {
        if( max<FABS(basisData[i]) ) max = FABS(basisData[i]);
    }

    return max;
}

#endif /* KEMFIELD_LINEARALGEBRA_CUH */
