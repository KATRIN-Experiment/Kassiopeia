#ifndef KEMFIELD_SOLIDANGLE_CUH
#define KEMFIELD_SOLIDANGLE_CUH

#include "kEMField_Triangle.cuh"
#include "kEMField_Rectangle.cuh"
#include "kEMField_VectorOperations.cuh"

// Functions for computing solid angle from Euler-Eriksson's formula as described in paper PIER 63, 243-278, 2006
// Author: Daniel Hilk

#define SOLIDANGLEMINDIST 1.e-15 /* for check if field point is on surface */

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE Tri_SolidAngle( const CU_TYPE* P, const CU_TYPE* data )
{
	CU_TYPE res = 0.;

    // corner points P0, P1 and P2
    const CU_TYPE triP0[3] = { data[2], data[3], data[4] };

    const CU_TYPE triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const CU_TYPE triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on triangle surface
	CU_TYPE n3[3];
	Tri_Normal( n3, data );

	// get triangle centroid
	CU_TYPE triCenter[3];
	Tri_Centroid( triCenter, data );

	// quantity h, magnitude corresponds to distance from field point to triangle plane
	const CU_TYPE h = (n3[0]*(P[0]-triCenter[0]))
			+ (n3[1]*(P[1]-triCenter[1]))
			+ (n3[2]*(P[2]-triCenter[2]));

	const CU_TYPE triMagCenterToP = SQRT(POW2(P[0]-triCenter[0]) + POW2(P[1]-triCenter[1]) + POW2(P[2]-triCenter[2]));

	if( triMagCenterToP <= SOLIDANGLEMINDIST )
		res = 2.*M_PI;
	else {
		// unit vectors of distances of corner points to field point in positive rotation order
		CU_TYPE triDistP0Unit[3];
		Compute_UnitVector( P, triP0, triDistP0Unit );
		CU_TYPE triDistP1Unit[3];
		Compute_UnitVector( P, triP1, triDistP1Unit );
		CU_TYPE triDistP2Unit[3];
		Compute_UnitVector( P, triP2, triDistP2Unit );

		const CU_TYPE x = 1.
				+ ( (triDistP0Unit[0]*triDistP1Unit[0])+(triDistP0Unit[1]*triDistP1Unit[1])+(triDistP0Unit[2]*triDistP1Unit[2]) )
				+ ( (triDistP0Unit[0]*triDistP2Unit[0])+(triDistP0Unit[1]*triDistP2Unit[1])+(triDistP0Unit[2]*triDistP2Unit[2]) )
				+ ( (triDistP1Unit[0]*triDistP2Unit[0])+(triDistP1Unit[1]*triDistP2Unit[1])+(triDistP1Unit[2]*triDistP2Unit[2]) );

		CU_TYPE a12[3];
		Compute_CrossProduct( triDistP1Unit, triDistP2Unit, a12 );

		const CU_TYPE y = FABS( ( (triDistP0Unit[0]*a12[0]) + (triDistP0Unit[1]*a12[1]) + (triDistP0Unit[2]*a12[2]) ) );

		res = FABS( 2.*atan2(y,x) );
	}

	if( h < 0. ) res *= -1.;

	return res;
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE Rect_SolidAngle( const CU_TYPE* P, const CU_TYPE* data )
{
    // corner points P0, P1, P2 and P3

    const CU_TYPE rectP0[3] = { data[2], data[3], data[4] };

    const CU_TYPE rectP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const CU_TYPE rectP2[3] = {
    		data[2] + (data[0]*data[5]) + (data[1]*data[8]),
    		data[3] + (data[0]*data[6]) + (data[1]*data[9]),
			data[4] + (data[0]*data[7]) + (data[1]*data[10]) }; // = fP0 + fN1*fA + fN2*fB

    const CU_TYPE rectP3[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// Computing solid angle from two triangles:
	// Triangle 1: P0 - P1 (N1) - P2 (N2)
	// Triangle 2: P2 - P3 (-N1) - P0 (-N2)

	CU_TYPE res = 0.;

	// get rectangle centroid

	CU_TYPE rectCenter[3];
	Rect_Centroid( rectCenter, data );

	// get perpendicular normal vector n3 on rectangle surface

	CU_TYPE rectN3[3];
	Rect_Normal( rectN3, data );

	// quantity h, magnitude corresponds to distance from field point to triangle plane

	const CU_TYPE h = (rectN3[0]*(P[0]-rectCenter[0])) + (rectN3[1]*(P[1]-rectCenter[1])) + (rectN3[2]*(P[2]-rectCenter[2]));

	// check if field point is on rectangle plane

	if( FABS(h) < SOLIDANGLEMINDIST ) {

		// check if field point is inside the rectangle plane

  		// line 1: P0 - P1, line 2: P0 - P3

		const CU_TYPE rectSide1[3] = {
  				data[0]*data[5],
				data[0]*data[6],
				data[0]*data[7]
  		};
  		const CU_TYPE rectSide2[3] = {
  				data[1]*data[8],
				data[1]*data[9],
				data[1]*data[10]
  		};

  		const CU_TYPE rectDistP[3] = {
  				data[2] - P[0],
				data[3] - P[1],
				data[4] - P[2]
  		};

    	// parameter lambda is needed for checking if point is on rectangle surface
    	// here: factor -1 for turning direction of reDistP

    	const CU_TYPE lineLambda1 = (-1) * ((rectDistP[0]*rectSide1[0])+(rectDistP[1]*rectSide1[1])+(rectDistP[2]*rectSide1[2])) / POW2(data[0]);
    	const CU_TYPE lineLambda2 = (-1) * ((rectDistP[0]*rectSide2[0])+(rectDistP[1]*rectSide2[1])+(rectDistP[2]*rectSide2[2])) / POW2(data[1]);

    	// field point is on rectangle plane

		if( lineLambda1>0. && lineLambda1<1. && lineLambda2>0. && lineLambda2<1. )
			res = 2.*M_PI;
		else // field point is in rectangle plane, but outside the surface
			res = 0.;
	}
	else {
		// unit vectors of distances of corner points to field point in positive rotation order

		CU_TYPE rectDistP0Unit[3];
		Compute_UnitVector( P, rectP0, rectDistP0Unit );
		CU_TYPE rectDistP1Unit[3];
		Compute_UnitVector( P, rectP1, rectDistP1Unit );
		CU_TYPE rectDistP2Unit[3];
		Compute_UnitVector( P, rectP2, rectDistP2Unit );
		CU_TYPE rectDistP3Unit[3];
		Compute_UnitVector( P, rectP3, rectDistP3Unit );

		const CU_TYPE x1 = 1.
				+ ( (rectDistP0Unit[0]*rectDistP1Unit[0])+(rectDistP0Unit[1]*rectDistP1Unit[1])+(rectDistP0Unit[2]*rectDistP1Unit[2]) )
				+ ( (rectDistP0Unit[0]*rectDistP2Unit[0])+(rectDistP0Unit[1]*rectDistP2Unit[1])+(rectDistP0Unit[2]*rectDistP2Unit[2]) )
				+ ( (rectDistP1Unit[0]*rectDistP2Unit[0])+(rectDistP1Unit[1]*rectDistP2Unit[1])+(rectDistP1Unit[2]*rectDistP2Unit[2]) );

		CU_TYPE a12[3];
		Compute_CrossProduct( rectDistP1Unit, rectDistP2Unit, a12 );

		const CU_TYPE y1 = FABS( (rectDistP0Unit[0]*a12[0])+(rectDistP0Unit[1]*a12[1])+(rectDistP0Unit[2]*a12[2]) );

		const CU_TYPE solidAngle1 = FABS( 2.*atan2(y1,x1) );

		const CU_TYPE x2 = 1.
				+ ( (rectDistP2Unit[0]*rectDistP3Unit[0])+(rectDistP2Unit[1]*rectDistP3Unit[1])+(rectDistP2Unit[2]*rectDistP3Unit[2]) )
				+ ( (rectDistP2Unit[0]*rectDistP0Unit[0])+(rectDistP2Unit[1]*rectDistP0Unit[1])+(rectDistP2Unit[2]*rectDistP0Unit[2]) )
				+ ( (rectDistP3Unit[0]*rectDistP0Unit[0])+(rectDistP3Unit[1]*rectDistP0Unit[1])+(rectDistP3Unit[2]*rectDistP0Unit[2]) );

		CU_TYPE a30[3];
		Compute_CrossProduct( rectDistP3Unit, rectDistP0Unit, a30 );

		const CU_TYPE y2 = FABS( (rectDistP2Unit[0]*a30[0])+(rectDistP2Unit[1]*a30[1])+(rectDistP2Unit[2]*a30[2]) );

		const CU_TYPE solidAngle2 = FABS( 2.*atan2(y2,x2) );

		res = solidAngle1 + solidAngle2;
	}

	if (h < 0.) res *= -1.;

	return res;
}



#endif /* KEMFIELD_SOLIDANGLE_CUH */
