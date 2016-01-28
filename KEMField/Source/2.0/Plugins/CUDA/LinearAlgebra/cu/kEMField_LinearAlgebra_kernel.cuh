#ifndef KEMFIELD_LINEARALGEBRA_KERNEL_CUH
#define KEMFIELD_LINEARALGEBRA_KERNEL_CUH

#include "kEMField_LinearAlgebra.cuh"

//______________________________________________________________________________

__global__ void GetMatrixElementKernel( int* ij,
        const int* boundaryInfo,
        const CU_TYPE* boundaryData,
        const short* shapeInfo,
        const CU_TYPE* shapeData,
        CU_TYPE* value )
{
    value[0] = LA_GetMatrixElement( ij[0],
            ij[1],
            boundaryInfo,
            boundaryData,
            shapeInfo,
            shapeData );
    return;
}

//______________________________________________________________________________

__global__ void GetVectorElementKernel( int* i,
        const int* boundaryInfo,
        const CU_TYPE* boundaryData,
        CU_TYPE* value )
{
    value[0] = LA_GetVectorElement( i[0],
            boundaryInfo,
            boundaryData );
    return;
}

//______________________________________________________________________________

__global__ void GetSolutionVectorElementKernel( int* i,
        CU_TYPE* q,
        CU_TYPE* value)
{
    value[0] = q[i[0]];
    return;
}

//______________________________________________________________________________

__global__ void GetMaximumVectorElementKernel( const int* boundaryInfo,
        const CU_TYPE* boundaryData,
        CU_TYPE* value )
{
    value[0] = LA_GetMaximumVectorElement(boundaryInfo,boundaryData);
    return;
}

//______________________________________________________________________________

__global__ void GetMaximumSolutionVectorElementKernel( const int* boundaryInfo,
        const CU_TYPE* basisData,
        CU_TYPE* value )
{
    value[0] = LA_GetMaximumSolutionVectorElement(boundaryInfo,basisData);
    return;
}

#endif /* KEMFIELD_LINEARALGEBRA_KERNEL_CUH */
