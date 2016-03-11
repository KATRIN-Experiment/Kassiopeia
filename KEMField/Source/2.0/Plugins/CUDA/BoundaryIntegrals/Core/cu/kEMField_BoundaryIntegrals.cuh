#ifndef KEMFIELD_BOUNDARYINTEGRALS_CUH
#define KEMFIELD_BOUNDARYINTEGRALS_CUH

#include "kEMField_cuda_defines.h"

#include "kEMField_Rectangle.cuh"
#include "kEMField_Triangle.cuh"
#include "kEMField_LineSegment.cuh"
// conic section code has been commented out since it leads to long compilation time
// and low speed increase in comparison to CPU code
//#include "kEMField_ConicSection.cuh"

//______________________________________________________________________________

__forceinline__ __device__ int BI_GetNumElements( const int* boundaryInfo )
{
    return boundaryInfo[0];
}

//______________________________________________________________________________

__forceinline__ __device__ int BI_GetNumBoundaries( const int* boundaryInfo )
{
    return boundaryInfo[1];
}

//______________________________________________________________________________

__forceinline__ __device__ int BI_GetBoundarySize( int iBoundary, const int* boundaryInfo )
{
    return boundaryInfo[2 + iBoundary*3];
}

//______________________________________________________________________________

__forceinline__ __device__ int BI_GetBoundaryType( int iBoundary, const int* boundaryInfo )
{
    return boundaryInfo[2 + iBoundary*3 + 1];
} 

//______________________________________________________________________________

__forceinline__ __device__ int BI_GetBoundaryStart(int iBoundary, const int* boundaryInfo)
{
    return boundaryInfo[2 + iBoundary*3 + 2];
}

//______________________________________________________________________________

__forceinline__ __device__ CU_TYPE BI_GetBoundaryValue( int iBoundary,
                                        const int* boundaryInfo,
                                        const CU_TYPE* boundaryData )
{
    if( BI_GetBoundaryType(iBoundary,boundaryInfo) == DIRICHLETBOUNDARY )
        return boundaryData[iBoundary*BOUNDARYSIZE];
    else
        return 0.;
} 

//______________________________________________________________________________

__forceinline__ __device__ int BI_GetBoundaryForElement( int element, const int* boundaryInfo )
{
    int k;
    int targetBoundary = -1;

    for (k=0;k<BI_GetNumBoundaries(boundaryInfo);k++) {
        if (element>=BI_GetBoundaryStart(k,boundaryInfo))
            targetBoundary++;
        else
            break;
    }

    return targetBoundary;
}

//______________________________________________________________________________

__forceinline__ __device__ void BI_Centroid( CU_TYPE* cen,
        const short* shapeType,
        const CU_TYPE* data )
{
    if( shapeType[0] == TRIANGLE )
        return Tri_Centroid(cen,data);
    else if( shapeType[0] == RECTANGLE )
        return Rect_Centroid(cen,data);
    else if( shapeType[0] == LINESEGMENT )
        return Line_Centroid(cen,data);
    //else if( shapeType[0] == CONICSECTION )
        //return ConicSection_Centroid(cen,data);

    cen[0] = 1.e30;
    cen[1] = 1.e30;
    cen[2] = 1.e30;
}

//______________________________________________________________________________

__forceinline__ __device__ void BI_Normal( CU_TYPE* norm,
        const short* shapeType,
        const CU_TYPE* data )
{
    if( shapeType[0] == TRIANGLE )
        return Tri_Normal(norm,data);
    else if( shapeType[0] == RECTANGLE )
        return Rect_Normal(norm,data);
    else if( shapeType[0] == LINESEGMENT )
        return Line_Normal(norm,data);
    //else if( shapeType[0] == CONICSECTION )
        //return ConicSection_Normal(norm,data);

    norm[0] = 0.;
    norm[1] = 0.;
    norm[2] = 0.;
}

#endif /* KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_CUH */
