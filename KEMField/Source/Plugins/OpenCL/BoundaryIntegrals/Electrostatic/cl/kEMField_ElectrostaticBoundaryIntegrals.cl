#ifndef KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_CL
#define KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_CL

#include "kEMField_opencl_defines.h"

#include "kEMField_BoundaryIntegrals.cl"

#ifdef RECTANGLE
#include "kEMField_ElectrostaticRectangle.cl"
#endif
#ifdef TRIANGLE
#include "kEMField_ElectrostaticTriangle.cl"
#endif
#ifdef LINESEGMENT
#include "kEMField_ElectrostaticLineSegment.cl"
#endif
#ifdef CONICSECTION
#include "kEMField_ElectrostaticConicSection.cl"
#endif


//______________________________________________________________________________

CL_TYPE EBI_Potential(const CL_TYPE* P,
		     __global const short* shapeType,
		     __global const CL_TYPE* data)
{
#ifdef TRIANGLE
  if (shapeType[0] == TRIANGLE)
    return ET_Potential(P,data);
#else
  if (false) {}
#endif
#ifdef RECTANGLE
  else if (shapeType[0] == RECTANGLE)
    return ER_Potential(P,data);
#endif
#ifdef LINESEGMENT
  else if (shapeType[0] == LINESEGMENT)
    return EL_Potential(P,data);
#endif
#ifdef CONICSECTION
  else if (shapeType[0] == CONICSECTION)
    return EC_Potential(P,data);
#endif

  return 0.;
}

//______________________________________________________________________________

CL_TYPE4 EBI_EField(const CL_TYPE* P,
		   __global const short* shapeType,
		   __global const CL_TYPE* data)
{
#ifdef TRIANGLE
  if (shapeType[0] == TRIANGLE)
    return ET_EField(P,data);
#else
  if (false) {}
#endif
#ifdef RECTANGLE
  else if (shapeType[0] == RECTANGLE)
    return ER_EField(P,data);
#endif
#ifdef LINESEGMENT
  else if (shapeType[0] == LINESEGMENT)
    return EL_EField(P,data);
#endif
#ifdef CONICSECTION
  else if (shapeType[0] == CONICSECTION)
    return EC_EField(P,data);
#endif

    return (CL_TYPE4)(0.,0.,0.,0.);
}


//______________________________________________________________________________

CL_TYPE4 EBI_EFieldAndPotential(const CL_TYPE* P,
		   __global const short* shapeType,
		   __global const CL_TYPE* data)
{
#ifdef TRIANGLE
  if (shapeType[0] == TRIANGLE)
    return ET_EFieldAndPotential(P,data);
#else
  if (false) {}
#endif
#ifdef RECTANGLE
  else if (shapeType[0] == RECTANGLE)
    return ER_EFieldAndPotential(P,data);
#endif
#ifdef LINESEGMENT
  else if (shapeType[0] == LINESEGMENT)
    return EL_EFieldAndPotential(P,data);
#endif
#ifdef CONICSECTION
  else if (shapeType[0] == CONICSECTION)
    return EC_EFieldAndPotential(P,data);
#endif

    return (CL_TYPE4)(0.,0.,0.,0.);
}

//______________________________________________________________________________

CL_TYPE BI_BoundaryIntegral(int iBoundary,
			    __global const int* boundaryInfo,
			    __global const CL_TYPE* boundaryData,
			    __global const short* shapeType_target,
			    __global const short* shapeType_source,
			    __global const CL_TYPE* data_target,
			    __global const CL_TYPE* data_source)
{
  CL_TYPE P_target[3];
  BI_Centroid(&P_target[0],shapeType_target,data_target);

  CL_TYPE4 eField = (CL_TYPE4)(0.,0.,0.,0.);
  CL_TYPE P_source[3];
  CL_TYPE N_target[3];
  CL_TYPE val;
  CL_TYPE dist2;

#ifdef DIRICHLETBOUNDARY
  if (BI_GetBoundaryType(iBoundary,boundaryInfo) == DIRICHLETBOUNDARY)
  {
    val = EBI_Potential(P_target,shapeType_source,data_source);
  }
#else
  if (false)
  {

  }
#endif
#ifdef NEUMANNBOUNDARY
  else // NEUMANN
  {
      BI_Centroid(&P_source[0],shapeType_source,data_source);
      BI_Normal(&N_target[0],shapeType_target,data_target);
      dist2 = ((P_target[0]-P_source[0])*(P_target[0]-P_source[0]) +
           (P_target[1]-P_source[1])*(P_target[1]-P_source[1]) +
           (P_target[2]-P_source[2])*(P_target[2]-P_source[2]));

      if( dist2 >= 1.e-24) {
        eField = EBI_EField(P_target,shapeType_source,data_source);
        val = eField.x*N_target[0] + eField.y*N_target[1] + eField.z*N_target[2];
      } else {
        // For planar Neumann elements (here: triangles and rectangles) the following formula
        // is valid and incorporates already the electric field 1./(2.*Eps0).
        // In case of conical (axially symmetric) Neumann elements this formula has to be modified.
        // Ferenc Glueck and Daniel Hilk, March 27th 2018
        val = ((1. + boundaryData[iBoundary*BOUNDARYSIZE])/(1. - boundaryData[iBoundary*BOUNDARYSIZE]))/(2.*M_EPS0);
      }
  }
#endif
  return val;
}

#endif /* KEMFIELD_ELECTROSTATICBOUNDARYINTEGRALS_CL */
