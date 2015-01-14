#ifndef KEMFIELD_BOUNDARYINTEGRALS_CL
#define KEMFIELD_BOUNDARYINTEGRALS_CL

#include "kEMField_defines.h"

#ifdef RECTANGLE
#include "kEMField_Rectangle.cl"
#endif
#ifdef TRIANGLE
#include "kEMField_Triangle.cl"
#endif
#ifdef LINESEGMENT
#include "kEMField_LineSegment.cl"
#endif
#ifdef CONICSECTION
#include "kEMField_ConicSection.cl"
#endif

//______________________________________________________________________________

int BI_GetNumElements(__global const int* boundaryInfo)
{
  return boundaryInfo[0];
}

//______________________________________________________________________________

int BI_GetNumBoundaries(__global const int* boundaryInfo)
{
  return boundaryInfo[1];
}

//______________________________________________________________________________

int BI_GetBoundarySize(int iBoundary,
		       __global const int* boundaryInfo)
{
  return boundaryInfo[2 + iBoundary*3];
}

//______________________________________________________________________________

int BI_GetBoundaryType(int iBoundary,
		       __global const int* boundaryInfo)
{
  return boundaryInfo[2 + iBoundary*3 + 1];
} 

//______________________________________________________________________________

int BI_GetBoundaryStart(int iBoundary,
			__global const int* boundaryInfo)
{
  return boundaryInfo[2 + iBoundary*3 + 2];
}

//______________________________________________________________________________

CL_TYPE BI_GetBoundaryValue(int iBoundary,
			    __global const int* boundaryInfo,
			    __global const CL_TYPE* boundaryData)
{
#ifdef DIRICHLETBOUNDARY
  if (BI_GetBoundaryType(iBoundary,boundaryInfo) == DIRICHLETBOUNDARY)
    return boundaryData[iBoundary*BOUNDARYSIZE];
  else
#endif
    return 0.;
} 

//______________________________________________________________________________

int BI_GetBoundaryForElement(int element,
			     __global const int* boundaryInfo)
{
  int k;
  int targetBoundary = -1;
  for (k=0;k<BI_GetNumBoundaries(boundaryInfo);k++)
  {
    if (element>=BI_GetBoundaryStart(k,boundaryInfo))
      targetBoundary++;
    else
      break;
  }

  return targetBoundary;
}

//______________________________________________________________________________

void BI_Centroid(CL_TYPE* cen,
		 __global const short* shapeType,
		 __global const CL_TYPE* data)
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

void BI_Normal(CL_TYPE* norm,
	       __global const short* shapeType,
	       __global const CL_TYPE* data)
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

#endif /* KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_CL */
