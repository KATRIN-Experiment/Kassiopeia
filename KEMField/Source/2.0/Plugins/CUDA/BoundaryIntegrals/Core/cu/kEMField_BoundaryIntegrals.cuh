#ifndef KEMFIELD_BOUNDARYINTEGRALS_CUH
#define KEMFIELD_BOUNDARYINTEGRALS_CUH

#include "kEMField_cuda_defines.h"

#ifdef RECTANGLE
#include "kEMField_Rectangle.cuh"
#endif
#ifdef TRIANGLE
#include "kEMField_Triangle.cuh"
#endif
#ifdef LINESEGMENT
#include "kEMField_LineSegment.cuh"
#endif
#ifdef CONICSECTION
#include "kEMField_ConicSection.cuh"
#endif

//______________________________________________________________________________

__forceinline__ __device__
int BI_GetNumElements( const int* boundaryInfo )
{
    return boundaryInfo[0];
}

//______________________________________________________________________________

__forceinline__ __device__
int BI_GetNumBoundaries( const int* boundaryInfo )
{
    return boundaryInfo[1];
}

//______________________________________________________________________________

__forceinline__ __device__
int BI_GetBoundarySize( int iBoundary, const int* boundaryInfo )
{
    return boundaryInfo[2 + iBoundary*3];
}

//______________________________________________________________________________

__forceinline__ __device__
int BI_GetBoundaryType( int iBoundary, const int* boundaryInfo )
{
    return boundaryInfo[2 + iBoundary*3 + 1];
} 

//______________________________________________________________________________

__forceinline__ __device__
int BI_GetBoundaryStart(int iBoundary, const int* boundaryInfo)
{
    return boundaryInfo[2 + iBoundary*3 + 2];
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE BI_GetBoundaryValue( int iBoundary,
                                        const int* boundaryInfo,
                                        const CU_TYPE* boundaryData )
{
#ifdef DIRICHLETBOUNDARY
    if( BI_GetBoundaryType(iBoundary,boundaryInfo) == DIRICHLETBOUNDARY )
        return boundaryData[iBoundary*BOUNDARYSIZE];
    else
#endif
    return 0.;
} 

//______________________________________________________________________________

__forceinline__ __device__
int BI_GetBoundaryForElement( int element, const int* boundaryInfo )
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

__forceinline__ __device__
void BI_Centroid( CU_TYPE* cen, const short* shapeType, const CU_TYPE* data )
{
#ifdef TRIANGLE
    if (shapeType[0] == TRIANGLE)
      return Tri_Centroid(cen,data);
#else
    if (false) { }
#endif
#ifdef RECTANGLE
    else if (shapeType[0] == RECTANGLE)
      return Rect_Centroid(cen,data);
#endif
#ifdef LINESEGMENT
    else if (shapeType[0] == LINESEGMENT)
      return Line_Centroid(cen,data);
#endif
#ifdef CONICSECTION
    else if (shapeType[0] == CONICSECTION)
      return ConicSection_Centroid(cen,data);
#endif

    cen[0] = 1.e30;
    cen[1] = 1.e30;
    cen[2] = 1.e30;
}

//______________________________________________________________________________

__forceinline__ __device__
void BI_Normal( CU_TYPE* norm, const short* shapeType, const CU_TYPE* data )
{
#ifdef TRIANGLE
    if (shapeType[0] == TRIANGLE)
      return Tri_Normal(norm,data);
#else
    if (false) {}
#endif
#ifdef RECTANGLE
    else if (shapeType[0] == RECTANGLE)
      return Rect_Normal(norm,data);
#endif
#ifdef LINESEGMENT
    else if (shapeType[0] == LINESEGMENT)
      return Line_Normal(norm,data);
#endif
#ifdef CONICSECTION
    else if (shapeType[0] == CONICSECTION)
      return ConicSection_Normal(norm,data);
#endif

    norm[0] = 0.;
    norm[1] = 0.;
    norm[2] = 0.;
}

#endif /* KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_CUH */
