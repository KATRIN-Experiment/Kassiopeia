#ifndef KEMFIELD_ELECTROSTATICNUMERICBOUNDARYINTEGRALS_CL
#define KEMFIELD_ELECTROSTATICNUMERICBOUNDARYINTEGRALS_CL

#include "kEMField_opencl_defines.h"

#include "kEMField_BoundaryIntegrals.cl"

#ifdef TRIANGLE
#define TRI_DRCUTOFFRWG 3.42 // distance ratio for switch from RWG to 33-point cubature
#define TRI_DRCUTOFFCUB33 30.4 // distance ratio for switch from 33-point cubature to 12-point cubature
#define TRI_DRCUTOFFCUB12 131.4 // distance ratio for switch from 12-point cubature to 7-point cubature
#include "kEMField_ElectrostaticCubatureTriangle_7Point.cl"
#include "kEMField_ElectrostaticCubatureTriangle_12Point.cl"
#include "kEMField_ElectrostaticCubatureTriangle_33Point.cl"
#include "kEMField_ElectrostaticRWGTriangle.cl"
#endif
#ifdef RECTANGLE
#define RECT_DRCUTOFFRWG 4.6 // distance ratio for switch from RWG to 33-point cubature
#define RECT_DRCUTOFFCUB33 44.6 // distance ratio for switch from 33-point cubature to 12-point cubature
#define RECT_DRCUTOFFCUB12 196.2 // distance ratio for switch from 12-point cubature to 7-point cubature
#include "kEMField_ElectrostaticCubatureRectangle_7Point.cl"
#include "kEMField_ElectrostaticCubatureRectangle_12Point.cl"
#include "kEMField_ElectrostaticCubatureRectangle_33Point.cl"
#include "kEMField_ElectrostaticRWGRectangle.cl"
#endif
#ifdef LINESEGMENT
#define LINE_DRCUTOFFANA 2. // distance ratio for switch from analytic integration to 16-node quadrature
#define LINE_DRCUTOFFQUAD16 27.7 // distance ratio for switch from 16-node quadrature to 4-node quadrature
#include "kEMField_ElectrostaticQuadratureLineSegment.cl"
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
	if (shapeType[0] == TRIANGLE) {
		// compute side length of triangle P1->P2

	    const CL_TYPE triP1[3] = {
	    		data[2] + (data[0]*data[5]),
	    		data[3] + (data[0]*data[6]),
				data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

	    const CL_TYPE triP2[3] = {
	    		data[2] + (data[1]*data[8]),
	    		data[3] + (data[1]*data[9]),
				data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	    const CL_TYPE triSideLengthP1P2
			= SQRT( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );

	    const CL_TYPE size = (data[0]+data[1]+triSideLengthP1P2)/3.;

		// get distance vector from field point to centroid of triangle

	    const CL_TYPE cen[3] = {
	    		data[2] + (data[0]*data[5] + data[1]*data[8])/3.,
				data[3] + (data[0]*data[6] + data[1]*data[9])/3.,
				data[4] + (data[0]*data[7] + data[1]*data[10])/3.};

		const CL_TYPE dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

		// magnitude of distance vector
		const CL_TYPE mag = SQRT(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

		// determine distance ratio (distance of field point to centroid over average triangle side length)
		const CL_TYPE tmpSize = 1./size;
		const CL_TYPE distanceRatio = mag*tmpSize;

        if( distanceRatio > TRI_DRCUTOFFCUB12 ) {
            return ET_Potential_Cub7P( P, data );
        }
		if( distanceRatio > TRI_DRCUTOFFCUB33 ) {
			return ET_Potential_Cub12P( P, data );
		}
		if( distanceRatio > TRI_DRCUTOFFRWG ) {
			return ET_Potential_Cub33P( P, data );
		}

		return ET_Potential( P, data );
  }
#else
	if (false) {}
#endif
#ifdef RECTANGLE
	else if (shapeType[0] == RECTANGLE) {
	    const CL_TYPE size = 0.5*(data[0]+data[1]);

		// get distance vector from field point to centroid of rectangle

	    const CL_TYPE cen[3] = {
	    		data[2] + 0.5*(data[0]*data[5] + data[1]*data[8]),
				data[3] + 0.5*(data[0]*data[6] + data[1]*data[9]),
				data[4] + 0.5*(data[0]*data[7] + data[1]*data[10])};

		const CL_TYPE dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

		// magnitude of distance vector
		const CL_TYPE mag = SQRT(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

		// determine distance ratio (distance of field point to centroid over average triangle side length)
		const CL_TYPE tmpSize = 1./size;
		const CL_TYPE distanceRatio = mag*tmpSize;

		if( distanceRatio > RECT_DRCUTOFFCUB12 ) {
			return ER_Potential_Cub7P( P, data );
		}
        if( distanceRatio > RECT_DRCUTOFFCUB33 ) {
            return ER_Potential_Cub12P( P, data );
        }
		if( distanceRatio > RECT_DRCUTOFFRWG ) {
			return ER_Potential_Cub33P( P, data );
		}

		return ER_Potential( P, data );
	}
#endif
#ifdef LINESEGMENT
	else if (shapeType[0] == LINESEGMENT) {
	    const CL_TYPE length = SQRT(
	    		((data[3]-data[0])*(data[3]-data[0]))
	    		+ ((data[4]-data[1])*(data[4]-data[1]))
				+ ((data[5]-data[2])*(data[5]-data[2])));

		// get distance vector from field point to centroid of line segment

	    const CL_TYPE cen[3] = {
	    		(data[0] + data[3])*0.5,
				(data[1] + data[4])*0.5,
				(data[2] + data[5])*0.5};

		const CL_TYPE dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

		// magnitude of distance vector
		const CL_TYPE mag = SQRT(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

		// determine distance ratio (distance of field point to centroid over line segment length)
		const CL_TYPE tmpSize = 1./length;
		const CL_TYPE distanceRatio = mag*tmpSize;

		if( distanceRatio > LINE_DRCUTOFFQUAD16 ) {
			return EL_Potential_Quad4N( P, data );
		}
		if( distanceRatio > LINE_DRCUTOFFANA ) {
			return EL_Potential_Quad16N( P, data );
		}

		return EL_Potential( P, data );
  }
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
  if (shapeType[0] == TRIANGLE) {
		// compute side length of triangle P1->P2

	    const CL_TYPE triP1[3] = {
	    		data[2] + (data[0]*data[5]),
	    		data[3] + (data[0]*data[6]),
				data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

	    const CL_TYPE triP2[3] = {
	    		data[2] + (data[1]*data[8]),
	    		data[3] + (data[1]*data[9]),
				data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	    const CL_TYPE triSideLengthP1P2
			= SQRT( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );

	    const CL_TYPE size = (data[0]+data[1]+triSideLengthP1P2)/3.;

		// get distance vector from field point to centroid of triangle

	    const CL_TYPE cen[3] = {
	    		data[2] + (data[0]*data[5] + data[1]*data[8])/3.,
				data[3] + (data[0]*data[6] + data[1]*data[9])/3.,
				data[4] + (data[0]*data[7] + data[1]*data[10])/3.};

		const CL_TYPE dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

		// magnitude of distance vector
		const CL_TYPE mag = SQRT(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

		// determine distance ratio (distance of field point to centroid over average triangle side length)
		const CL_TYPE tmpSize = 1./size;
		const CL_TYPE distanceRatio = mag*tmpSize;

		if( distanceRatio > TRI_DRCUTOFFCUB12 ) {
            return ET_EField_Cub7P( P, data );
        }
		if( distanceRatio > TRI_DRCUTOFFCUB33 ) {
			return ET_EField_Cub12P( P, data );
		}
		if( distanceRatio > TRI_DRCUTOFFRWG ) {
			return ET_EField_Cub33P( P, data );
		}

		return ET_EField( P, data );
  }
#else
  if (false) {}
#endif
#ifdef RECTANGLE
  else if (shapeType[0] == RECTANGLE) {
	    const CL_TYPE size = 0.5*(data[0]+data[1]);

		// get distance vector from field point to centroid of rectangle

	    const CL_TYPE cen[3] = {
	    		data[2] + 0.5*(data[0]*data[5] + data[1]*data[8]),
				data[3] + 0.5*(data[0]*data[6] + data[1]*data[9]),
				data[4] + 0.5*(data[0]*data[7] + data[1]*data[10])};

		const CL_TYPE dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

		// magnitude of distance vector
		const CL_TYPE mag = SQRT(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

		// determine distance ratio (distance of field point to centroid over average triangle side length)
		const CL_TYPE tmpSize = 1./size;
		const CL_TYPE distanceRatio = mag*tmpSize;

        if( distanceRatio > RECT_DRCUTOFFCUB12 ) {
            return ER_EField_Cub7P( P, data );
        }
		if( distanceRatio > RECT_DRCUTOFFCUB33 ) {
			return ER_EField_Cub12P( P, data );
		}
		if( distanceRatio > RECT_DRCUTOFFRWG ) {
			return ER_EField_Cub33P( P, data );
		}

		return ER_EField( P, data );
  }
#endif
#ifdef LINESEGMENT
  else if (shapeType[0] == LINESEGMENT) {
	    const CL_TYPE length = SQRT(
	    		((data[3]-data[0])*(data[3]-data[0]))
	    		+ ((data[4]-data[1])*(data[4]-data[1]))
				+ ((data[5]-data[2])*(data[5]-data[2])));

		// get distance vector from field point to centroid of line segment

	    const CL_TYPE cen[3] = {
	    		(data[0] + data[3])*0.5,
				(data[1] + data[4])*0.5,
				(data[2] + data[5])*0.5};

		const CL_TYPE dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

		// magnitude of distance vector
		const CL_TYPE mag = SQRT(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

		// determine distance ratio (distance of field point to centroid over line segment length)
		const CL_TYPE tmpSize = 1./length;
		const CL_TYPE distanceRatio = mag*tmpSize;

		if( distanceRatio > LINE_DRCUTOFFQUAD16 ) {
			return EL_EField_Quad4N( P, data );
		}
		if( distanceRatio > LINE_DRCUTOFFANA ) {
			return EL_EField_Quad16N( P, data );
		}

		return EL_EField( P, data );
  }
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
  if (shapeType[0] == TRIANGLE) {
		// compute side length of triangle P1->P2

	    const CL_TYPE triP1[3] = {
	    		data[2] + (data[0]*data[5]),
	    		data[3] + (data[0]*data[6]),
				data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

	    const CL_TYPE triP2[3] = {
	    		data[2] + (data[1]*data[8]),
	    		data[3] + (data[1]*data[9]),
				data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	    const CL_TYPE triSideLengthP1P2
			= SQRT( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );

	    const CL_TYPE size = (data[0]+data[1]+triSideLengthP1P2)/3.;

		// get distance vector from field point to centroid of triangle

	    const CL_TYPE cen[3] = {
	    		data[2] + (data[0]*data[5] + data[1]*data[8])/3.,
				data[3] + (data[0]*data[6] + data[1]*data[9])/3.,
				data[4] + (data[0]*data[7] + data[1]*data[10])/3.};

		const CL_TYPE dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

		// magnitude of distance vector
		const CL_TYPE mag = SQRT(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

		// determine distance ratio (distance of field point to centroid over average triangle side length)
		const CL_TYPE tmpSize = 1./size;
		const CL_TYPE distanceRatio = mag*tmpSize;

        if( distanceRatio > TRI_DRCUTOFFCUB12 ) {
            return ET_EFieldAndPotential_Cub7P( P, data );
        }
		if( distanceRatio > TRI_DRCUTOFFCUB33 ) {
			return ET_EFieldAndPotential_Cub12P( P, data );
		}
		if( distanceRatio > TRI_DRCUTOFFRWG ) {
			return ET_EFieldAndPotential_Cub33P( P, data );
		}

		return ET_EFieldAndPotential( P, data );
}
#else
  if (false) {}
#endif
#ifdef RECTANGLE
  else if (shapeType[0] == RECTANGLE) {
	    const CL_TYPE size = 0.5*(data[0]+data[1]);

		// get distance vector from field point to centroid of rectangle

	    const CL_TYPE cen[3] = {
	    		data[2] + 0.5*(data[0]*data[5] + data[1]*data[8]),
				data[3] + 0.5*(data[0]*data[6] + data[1]*data[9]),
				data[4] + 0.5*(data[0]*data[7] + data[1]*data[10])};

		const CL_TYPE dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

		// magnitude of distance vector
		const CL_TYPE mag = SQRT(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

		// determine distance ratio (distance of field point to centroid over average triangle side length)
		const CL_TYPE tmpSize = 1./size;
		const CL_TYPE distanceRatio = mag*tmpSize;

        if( distanceRatio > RECT_DRCUTOFFCUB12 ) {
            return ER_EFieldAndPotential_Cub7P( P, data );
        }
		if( distanceRatio > RECT_DRCUTOFFCUB33 ) {
			return ER_EFieldAndPotential_Cub12P( P, data );
		}
		if( distanceRatio > RECT_DRCUTOFFRWG ) {
			return ER_EFieldAndPotential_Cub33P( P, data );
		}

		return ER_EFieldAndPotential( P, data );
  }
#endif
#ifdef LINESEGMENT
  else if (shapeType[0] == LINESEGMENT) {
	    const CL_TYPE length = SQRT(
	    		((data[3]-data[0])*(data[3]-data[0]))
	    		+ ((data[4]-data[1])*(data[4]-data[1]))
				+ ((data[5]-data[2])*(data[5]-data[2])));

		// get distance vector from field point to centroid of line segment

	    const CL_TYPE cen[3] = {
	    		(data[0] + data[3])*0.5,
				(data[1] + data[4])*0.5,
				(data[2] + data[5])*0.5};

		const CL_TYPE dist[3] = { cen[0]-P[0], cen[1]-P[1], cen[2]-P[2] };

		// magnitude of distance vector
		const CL_TYPE mag = SQRT(POW2(dist[0]) + POW2(dist[1]) + POW2(dist[2]));

		// determine distance ratio (distance of field point to centroid over line segment length)
		const CL_TYPE tmpSize = 1./length;
		const CL_TYPE distanceRatio = mag*tmpSize;

		if( distanceRatio > LINE_DRCUTOFFQUAD16 ) {
			return EL_EFieldAndPotential_Quad4N( P, data );
		}
		if( distanceRatio > LINE_DRCUTOFFANA ) {
			return EL_EFieldAndPotential_Quad16N( P, data );
		}

		return EL_EFieldAndPotential( P, data );
  }
#endif
#ifdef CONICSECTION
  else if (shapeType[0] == CONICSECTION) {
    return EC_EFieldAndPotential( P, data );
  }
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
    eField = EBI_EField(P_target,shapeType_source,data_source);
    BI_Centroid(&P_source[0],shapeType_source,data_source);
    BI_Normal(&N_target[0],shapeType_target,data_target);

    val = eField.x*N_target[0] + eField.y*N_target[1] + eField.z*N_target[2];
    dist2 = ((P_target[0]-P_source[0])*(P_target[0]-P_source[0]) +
	     (P_target[1]-P_source[1])*(P_target[1]-P_source[1]) +
	     (P_target[2]-P_source[2])*(P_target[2]-P_source[2]));
    if (dist2<1.e-24)
    {
      val = val*((1. + boundaryData[iBoundary*BOUNDARYSIZE])/
		 (1. - boundaryData[iBoundary*BOUNDARYSIZE]));
    }
  }
#endif
  return val;
}

#endif /* KEMFIELD_ELECTROSTATICNUMERICBOUNDARYINTEGRALS_CL */
