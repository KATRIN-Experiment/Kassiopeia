#ifndef KEMFIELD_ELECTROSTATICRWGRECTANGLE_CUH
#define KEMFIELD_ELECTROSTATICRWGRECTANGLE_CUH

// CUDA kernel for integration of rectangle surface for electrostatics in RWG basis
// Detailed information on the implementation can be found in the CPU code,
// class 'KElectrostaticRWGRectangleIntegrator'.
// Author: Daniel Hilk
//
// Recommended thread block sizes by CUDA Occupancy API for NVIDIA Tesla K40c:
// * ER_Potential: 384
// * ER_EField: 512
// * ER_EFieldAndPotential: 512
// Recommended thread block sizes are equal for standard and 'KEMFIELD_CUFASTRWG' option.

#include "kEMField_Rectangle.cuh"
#include "kEMField_VectorOperations.cuh"
#include "kEMField_SolidAngle.cuh"

// Rectangle geometry definition (as defined by the streamers in KRectangle.hh):
//
// data[0]:     A
// data[1]:     B
// data[2..4]:  P0[0..2]
// data[5..7]:  N1[0..2]
// data[8..10]: N2[0..2]

#define MINDISTANCETORECTLINE 1.E-14
#define CORRECTIONRECTN3 1.E-7 /* step in N3 direction if field point is on edge */
#define RECTLOGARGQUOTIENT 1.E-6 /* limit for Taylor expansion if field point is on line (=dist/sM) */

//______________________________________________________________________________

#ifndef KEMFIELD_CUFASTRWG
__forceinline__ __device__
CU_TYPE ER_LogArgTaylor( const CU_TYPE sMin, const CU_TYPE dist )
{
	CU_TYPE quotient = FABS(dist/sMin);
	if( quotient < 1.e-14 ) quotient = 1.e-14;
	return 0.5*FABS(sMin)*POW2(quotient);
}
#endif /* KEMFIELD_CUFASTRWG */

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE ER_IqLPotential( const unsigned short countCross, const unsigned short lineIndex,
		const CU_TYPE dist, const CU_TYPE* P, const CU_TYPE* data )
{
    // corner points P0, P1, P2 and P3

    const CU_TYPE rectP0[3] = {
    		data[2], data[3], data[4] };

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

	// get perpendicular normal vector n3 on rectangle surface

    CU_TYPE n3[3];
    Rect_Normal( n3, data );

    // side line unit vectors

    const CU_TYPE rectAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    const CU_TYPE rectAlongSideP1P2Unit[3] = {data[8], data[9], data[10]}; // = N2
    const CU_TYPE rectAlongSideP2P3Unit[3] = {-data[5], -data[6], -data[7]}; // = -N1
    const CU_TYPE rectAlongSideP3P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

    // center point of each side to field point

    const CU_TYPE e0[3] = {
    		(0.5*data[0]*data[5]) + rectP0[0],
			(0.5*data[0]*data[6]) + rectP0[1],
			(0.5*data[0]*data[7]) + rectP0[2] };

    const CU_TYPE e1[3] = {
    		(0.5*data[1]*data[8]) + rectP1[0],
			(0.5*data[1]*data[9]) + rectP1[1],
			(0.5*data[1]*data[10]) + rectP1[2] };

    const CU_TYPE e2[3] = {
    		(-0.5*data[0]*data[5]) + rectP2[0],
			(-0.5*data[0]*data[6]) + rectP2[1],
			(-0.5*data[0]*data[7]) + rectP2[2] };

    const CU_TYPE e3[3] = {
    		(-0.5*data[1]*data[8]) + rectP3[0],
			(-0.5*data[1]*data[9]) + rectP3[1],
			(-0.5*data[1]*data[10]) + rectP3[2] };

    // outward pointing vector m, perpendicular to side lines

    CU_TYPE m0[3];
    Compute_CrossProduct( rectAlongSideP0P1Unit, n3, m0 );
    CU_TYPE m1[3];
    Compute_CrossProduct( rectAlongSideP1P2Unit, n3, m1 );
    CU_TYPE m2[3];
    Compute_CrossProduct( rectAlongSideP2P3Unit, n3, m2 );
    CU_TYPE m3[3];
    Compute_CrossProduct( rectAlongSideP3P0Unit, n3, m3 );

    // size t

    const CU_TYPE t0 = (m0[0]*(P[0]-e0[0])) + (m0[1]*(P[1]-e0[1])) + (m0[2]*(P[2]-e0[2]));
    const CU_TYPE t1 = (m1[0]*(P[0]-e1[0])) + (m1[1]*(P[1]-e1[1])) + (m1[2]*(P[2]-e1[2]));
    const CU_TYPE t2 = (m2[0]*(P[0]-e2[0])) + (m2[1]*(P[1]-e2[1])) + (m2[2]*(P[2]-e2[2]));
    const CU_TYPE t3 = (m3[0]*(P[0]-e3[0])) + (m3[1]*(P[1]-e3[1])) + (m3[2]*(P[2]-e3[2]));

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const CU_TYPE rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const CU_TYPE rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const CU_TYPE rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const CU_TYPE rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };

   	const CU_TYPE rectDistP0Mag = SQRT( POW2(rectDistP0[0]) + POW2(rectDistP0[1]) + POW2(rectDistP0[2]) );
	const CU_TYPE rectDistP1Mag = SQRT( POW2(rectDistP1[0]) + POW2(rectDistP1[1]) + POW2(rectDistP1[2]) );
	const CU_TYPE rectDistP2Mag = SQRT( POW2(rectDistP2[0]) + POW2(rectDistP2[1]) + POW2(rectDistP2[2]) );
	const CU_TYPE rectDistP3Mag = SQRT( POW2(rectDistP3[0]) + POW2(rectDistP3[1]) + POW2(rectDistP3[2]) );

    // evaluation of line integral

    // 0 //

	CU_TYPE logArgNom = 0.;
	CU_TYPE logArgDenom = 0.;
	CU_TYPE iL = 0.;

	CU_TYPE rM = rectDistP0Mag;
	CU_TYPE rP = rectDistP1Mag;
	CU_TYPE sM = (rectDistP0[0]*rectAlongSideP0P1Unit[0]) + (rectDistP0[1]*rectAlongSideP0P1Unit[1]) + (rectDistP0[2]*rectAlongSideP0P1Unit[2]);
	CU_TYPE sP = (rectDistP1[0]*rectAlongSideP0P1Unit[0]) + (rectDistP1[1]*rectAlongSideP0P1Unit[1]) + (rectDistP1[2]*rectAlongSideP0P1Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==0) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			iL += ( t0 * (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) )) );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 0 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		iL += ( t0 * (LOG(logArgNom)-LOG(logArgDenom)) );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	// 1 //

	rM = rectDistP1Mag;
	rP = rectDistP2Mag;
	sM = (rectDistP1[0]*rectAlongSideP1P2Unit[0]) + (rectDistP1[1]*rectAlongSideP1P2Unit[1]) + (rectDistP1[2]*rectAlongSideP1P2Unit[2]);
	sP = (rectDistP2[0]*rectAlongSideP1P2Unit[0]) + (rectDistP2[1]*rectAlongSideP1P2Unit[1]) + (rectDistP2[2]*rectAlongSideP1P2Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==1) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			iL += ( t1 * (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) )) );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 1 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		iL += ( t1 * (LOG(logArgNom)-LOG(logArgDenom)) );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	// 2 //

	rM = rectDistP2Mag;
	rP = rectDistP3Mag;
	sM = (rectDistP2[0]*rectAlongSideP2P3Unit[0]) + (rectDistP2[1]*rectAlongSideP2P3Unit[1]) + (rectDistP2[2]*rectAlongSideP2P3Unit[2]);
	sP = (rectDistP3[0]*rectAlongSideP2P3Unit[0]) + (rectDistP3[1]*rectAlongSideP2P3Unit[1]) + (rectDistP3[2]*rectAlongSideP2P3Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==2) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			iL += ( t2 * (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) )) );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 2 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		iL += ( t2 * (LOG(logArgNom)-LOG(logArgDenom)) );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	// 3 //

	rM = rectDistP3Mag;
	rP = rectDistP0Mag;
	sM = (rectDistP3[0]*rectAlongSideP3P0Unit[0]) + (rectDistP3[1]*rectAlongSideP3P0Unit[1]) + (rectDistP3[2]*rectAlongSideP3P0Unit[2]);
	sP = (rectDistP0[0]*rectAlongSideP3P0Unit[0]) + (rectDistP0[1]*rectAlongSideP3P0Unit[1]) + (rectDistP0[2]*rectAlongSideP3P0Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==3) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			iL += ( t3 * (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) )) );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 3 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		iL += ( t3 * (LOG(logArgNom)-LOG(logArgDenom)) );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	return iL;
}


//______________________________________________________________________________


__forceinline__ __device__
CU_TYPE4 ER_IqLField( const unsigned short countCross, const unsigned short lineIndex, const CU_TYPE dist, const CU_TYPE* P, const CU_TYPE* data )
{
    // corner points P0, P1, P2 and P3

    const CU_TYPE rectP0[3] = {
    		data[2], data[3], data[4] };

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

	// get perpendicular normal vector n3 on rectangle surface

    CU_TYPE n3[3];
    Rect_Normal( n3, data );

    // side line unit vectors

    CU_TYPE rectAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    CU_TYPE rectAlongSideP1P2Unit[3] = {data[8], data[9], data[10]}; // = N2
    CU_TYPE rectAlongSideP2P3Unit[3] = {-data[5], -data[6], -data[7]}; // = -N1
    CU_TYPE rectAlongSideP3P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

    // outward pointing vector m, perpendicular to side lines

    CU_TYPE m0[3];
    Compute_CrossProduct( rectAlongSideP0P1Unit, n3, m0 );
    CU_TYPE m1[3];
    Compute_CrossProduct( rectAlongSideP1P2Unit, n3, m1 );
    CU_TYPE m2[3];
    Compute_CrossProduct( rectAlongSideP2P3Unit, n3, m2 );
    CU_TYPE m3[3];
    Compute_CrossProduct( rectAlongSideP3P0Unit, n3, m3 );

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const CU_TYPE rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const CU_TYPE rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const CU_TYPE rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const CU_TYPE rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };

   	const CU_TYPE rectDistP0Mag = SQRT( POW2(rectDistP0[0]) + POW2(rectDistP0[1]) + POW2(rectDistP0[2]) );
	const CU_TYPE rectDistP1Mag = SQRT( POW2(rectDistP1[0]) + POW2(rectDistP1[1]) + POW2(rectDistP1[2]) );
	const CU_TYPE rectDistP2Mag = SQRT( POW2(rectDistP2[0]) + POW2(rectDistP2[1]) + POW2(rectDistP2[2]) );
	const CU_TYPE rectDistP3Mag = SQRT( POW2(rectDistP3[0]) + POW2(rectDistP3[1]) + POW2(rectDistP3[2]) );

    // evaluation of line integral

    // 0 //

	CU_TYPE logArgNom = 0.;
	CU_TYPE logArgDenom = 0.;
	CU_TYPE iL[3] = {0., 0., 0.};
	CU_TYPE tmpScalar = 0.;

	CU_TYPE rM = rectDistP0Mag;
	CU_TYPE rP = rectDistP1Mag;
	CU_TYPE sM = (rectDistP0[0]*rectAlongSideP0P1Unit[0]) + (rectDistP0[1]*rectAlongSideP0P1Unit[1]) + (rectDistP0[2]*rectAlongSideP0P1Unit[2]);
	CU_TYPE sP = (rectDistP1[0]*rectAlongSideP0P1Unit[0]) + (rectDistP1[1]*rectAlongSideP0P1Unit[1]) + (rectDistP1[2]*rectAlongSideP0P1Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==0) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) ));
			iL[0] += ( m0[0] * tmpScalar );
			iL[1] += ( m0[1] * tmpScalar );
			iL[2] += ( m0[2] * tmpScalar );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 0 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		tmpScalar = (LOG( logArgNom )-LOG( logArgDenom ));
		iL[0] += ( m0[0] * tmpScalar );
		iL[1] += ( m0[1] * tmpScalar );
		iL[2] += ( m0[2] * tmpScalar );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	// 1 //

	rM = rectDistP1Mag;
	rP = rectDistP2Mag;
	sM = (rectDistP1[0]*rectAlongSideP1P2Unit[0]) + (rectDistP1[1]*rectAlongSideP1P2Unit[1]) + (rectDistP1[2]*rectAlongSideP1P2Unit[2]);
	sP = (rectDistP2[0]*rectAlongSideP1P2Unit[0]) + (rectDistP2[1]*rectAlongSideP1P2Unit[1]) + (rectDistP2[2]*rectAlongSideP1P2Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==1) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) ));
			iL[0] += ( m1[0] * tmpScalar );
			iL[1] += ( m1[1] * tmpScalar );
			iL[2] += ( m1[2] * tmpScalar );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 1 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		tmpScalar = (LOG( logArgNom )-LOG( logArgDenom ));
		iL[0] += ( m1[0] * tmpScalar );
		iL[1] += ( m1[1] * tmpScalar );
		iL[2] += ( m1[2] * tmpScalar );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	// 2 //

	rM = rectDistP2Mag;
	rP = rectDistP3Mag;
	sM = (rectDistP2[0]*rectAlongSideP2P3Unit[0]) + (rectDistP2[1]*rectAlongSideP2P3Unit[1]) + (rectDistP2[2]*rectAlongSideP2P3Unit[2]);
	sP = (rectDistP3[0]*rectAlongSideP2P3Unit[0]) + (rectDistP3[1]*rectAlongSideP2P3Unit[1]) + (rectDistP3[2]*rectAlongSideP2P3Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==2) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) ));
			iL[0] += ( m2[0] * tmpScalar );
			iL[1] += ( m2[1] * tmpScalar );
			iL[2] += ( m2[2] * tmpScalar );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 2 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		tmpScalar = (LOG( logArgNom )-LOG( logArgDenom ));
		iL[0] += ( m2[0] * tmpScalar );
		iL[1] += ( m2[1] * tmpScalar );
		iL[2] += ( m2[2] * tmpScalar );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	// 3 //

	rM = rectDistP3Mag;
	rP = rectDistP0Mag;
	sM = (rectDistP3[0]*rectAlongSideP3P0Unit[0]) + (rectDistP3[1]*rectAlongSideP3P0Unit[1]) + (rectDistP3[2]*rectAlongSideP3P0Unit[2]);
	sP = (rectDistP0[0]*rectAlongSideP3P0Unit[0]) + (rectDistP0[1]*rectAlongSideP3P0Unit[1]) + (rectDistP0[2]*rectAlongSideP3P0Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==3) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) ));
			iL[0] += ( m3[0] * tmpScalar );
			iL[1] += ( m3[1] * tmpScalar );
			iL[2] += ( m3[2] * tmpScalar );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 3 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		tmpScalar = (LOG( logArgNom )-LOG( logArgDenom ));
		iL[0] += ( m3[0] * tmpScalar );
		iL[1] += ( m3[1] * tmpScalar );
		iL[2] += ( m3[2] * tmpScalar );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	return MAKECU4( iL[0], iL[1], iL[2], 0. );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ER_IqLFieldAndPotential( const unsigned short countCross, const unsigned short lineIndex,
		const CU_TYPE dist, const CU_TYPE* P, const CU_TYPE* data )
{
    // corner points P0, P1, P2 and P3

    const CU_TYPE rectP0[3] = {
    		data[2], data[3], data[4] };

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

	// get perpendicular normal vector n3 on rectangle surface

    CU_TYPE n3[3];
    Rect_Normal( n3, data );

    // side line unit vectors

    CU_TYPE rectAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    CU_TYPE rectAlongSideP1P2Unit[3] = {data[8], data[9], data[10]}; // = N2
    CU_TYPE rectAlongSideP2P3Unit[3] = {-data[5], -data[6], -data[7]}; // = -N1
    CU_TYPE rectAlongSideP3P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

    // center point of each side to field point

    const CU_TYPE e0[3] = {
    		(0.5*data[0]*data[5]) + rectP0[0],
			(0.5*data[0]*data[6]) + rectP0[1],
			(0.5*data[0]*data[7]) + rectP0[2] };

    const CU_TYPE e1[3] = {
    		(0.5*data[1]*data[8]) + rectP1[0],
			(0.5*data[1]*data[9]) + rectP1[1],
			(0.5*data[1]*data[10]) + rectP1[2] };

    const CU_TYPE e2[3] = {
    		(-0.5*data[0]*data[5]) + rectP2[0],
			(-0.5*data[0]*data[6]) + rectP2[1],
			(-0.5*data[0]*data[7]) + rectP2[2] };

    const CU_TYPE e3[3] = {
    		(-0.5*data[1]*data[8]) + rectP3[0],
			(-0.5*data[1]*data[9]) + rectP3[1],
			(-0.5*data[1]*data[10]) + rectP3[2] };

    // outward pointing vector m, perpendicular to side lines

    CU_TYPE m0[3];
    Compute_CrossProduct( rectAlongSideP0P1Unit, n3, m0 );
    CU_TYPE m1[3];
    Compute_CrossProduct( rectAlongSideP1P2Unit, n3, m1 );
    CU_TYPE m2[3];
    Compute_CrossProduct( rectAlongSideP2P3Unit, n3, m2 );
    CU_TYPE m3[3];
    Compute_CrossProduct( rectAlongSideP3P0Unit, n3, m3 );

    // size t

    const CU_TYPE t0 = (m0[0]*(P[0]-e0[0])) + (m0[1]*(P[1]-e0[1])) + (m0[2]*(P[2]-e0[2]));
    const CU_TYPE t1 = (m1[0]*(P[0]-e1[0])) + (m1[1]*(P[1]-e1[1])) + (m1[2]*(P[2]-e1[2]));
    const CU_TYPE t2 = (m2[0]*(P[0]-e2[0])) + (m2[1]*(P[1]-e2[1])) + (m2[2]*(P[2]-e2[2]));
    const CU_TYPE t3 = (m3[0]*(P[0]-e3[0])) + (m3[1]*(P[1]-e3[1])) + (m3[2]*(P[2]-e3[2]));

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const CU_TYPE rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const CU_TYPE rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const CU_TYPE rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const CU_TYPE rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };

   	const CU_TYPE rectDistP0Mag = SQRT( POW2(rectDistP0[0]) + POW2(rectDistP0[1]) + POW2(rectDistP0[2]) );
	const CU_TYPE rectDistP1Mag = SQRT( POW2(rectDistP1[0]) + POW2(rectDistP1[1]) + POW2(rectDistP1[2]) );
	const CU_TYPE rectDistP2Mag = SQRT( POW2(rectDistP2[0]) + POW2(rectDistP2[1]) + POW2(rectDistP2[2]) );
	const CU_TYPE rectDistP3Mag = SQRT( POW2(rectDistP3[0]) + POW2(rectDistP3[1]) + POW2(rectDistP3[2]) );

    // evaluation of line integral

    // 0 //

	CU_TYPE logArgNom = 0.;
	CU_TYPE logArgDenom = 0.;
	CU_TYPE iLField[3] = {0., 0., 0.};
	CU_TYPE iLPhi = 0.;
	CU_TYPE tmpScalar = 0.;

	CU_TYPE rM = rectDistP0Mag;
	CU_TYPE rP = rectDistP1Mag;
	CU_TYPE sM = (rectDistP0[0]*rectAlongSideP0P1Unit[0]) + (rectDistP0[1]*rectAlongSideP0P1Unit[1]) + (rectDistP0[2]*rectAlongSideP0P1Unit[2]);
	CU_TYPE sP = (rectDistP1[0]*rectAlongSideP0P1Unit[0]) + (rectDistP1[1]*rectAlongSideP0P1Unit[1]) + (rectDistP1[2]*rectAlongSideP0P1Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==0) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) ));
			iLField[0] += ( m0[0] * tmpScalar );
			iLField[1] += ( m0[1] * tmpScalar );
			iLField[2] += ( m0[2] * tmpScalar );
			iLPhi += ( t0 * tmpScalar );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 0 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		tmpScalar = (LOG( logArgNom )-LOG( logArgDenom ));
		iLField[0] += ( m0[0] * tmpScalar );
		iLField[1] += ( m0[1] * tmpScalar );
		iLField[2] += ( m0[2] * tmpScalar );
		iLPhi += ( t0 * tmpScalar );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	// 1 //

	rM = rectDistP1Mag;
	rP = rectDistP2Mag;
	sM = (rectDistP1[0]*rectAlongSideP1P2Unit[0]) + (rectDistP1[1]*rectAlongSideP1P2Unit[1]) + (rectDistP1[2]*rectAlongSideP1P2Unit[2]);
	sP = (rectDistP2[0]*rectAlongSideP1P2Unit[0]) + (rectDistP2[1]*rectAlongSideP1P2Unit[1]) + (rectDistP2[2]*rectAlongSideP1P2Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==1) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) ));
			iLField[0] += ( m1[0] * tmpScalar );
			iLField[1] += ( m1[1] * tmpScalar );
			iLField[2] += ( m1[2] * tmpScalar );
			iLPhi += ( t1 * tmpScalar );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 1 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		tmpScalar = (LOG( logArgNom )-LOG( logArgDenom ));
		iLField[0] += ( m1[0] * tmpScalar );
		iLField[1] += ( m1[1] * tmpScalar );
		iLField[2] += ( m1[2] * tmpScalar );
		iLPhi += ( t1 * tmpScalar );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	// 2 //

	rM = rectDistP2Mag;
	rP = rectDistP3Mag;
	sM = (rectDistP2[0]*rectAlongSideP2P3Unit[0]) + (rectDistP2[1]*rectAlongSideP2P3Unit[1]) + (rectDistP2[2]*rectAlongSideP2P3Unit[2]);
	sP = (rectDistP3[0]*rectAlongSideP2P3Unit[0]) + (rectDistP3[1]*rectAlongSideP2P3Unit[1]) + (rectDistP3[2]*rectAlongSideP2P3Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==2) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) ));
			iLField[0] += ( m2[0] * tmpScalar );
			iLField[1] += ( m2[1] * tmpScalar );
			iLField[2] += ( m2[2] * tmpScalar );
			iLPhi += ( t2 * tmpScalar );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 2 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		tmpScalar = (LOG( logArgNom )-LOG( logArgDenom ));
		iLField[0] += ( m2[0] * tmpScalar );
		iLField[1] += ( m2[1] * tmpScalar );
		iLField[2] += ( m2[2] * tmpScalar );
		iLPhi += ( t2 * tmpScalar );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	// 3 //

	rM = rectDistP3Mag;
	rP = rectDistP0Mag;
	sM = (rectDistP3[0]*rectAlongSideP3P0Unit[0]) + (rectDistP3[1]*rectAlongSideP3P0Unit[1]) + (rectDistP3[2]*rectAlongSideP3P0Unit[2]);
	sP = (rectDistP0[0]*rectAlongSideP3P0Unit[0]) + (rectDistP0[1]*rectAlongSideP3P0Unit[1]) + (rectDistP0[2]*rectAlongSideP3P0Unit[2]);

#ifndef KEMFIELD_CUFASTRWG
	if ( (countCross==1) && (lineIndex==3) ) {
		if( FABS(dist/sM) < RECTLOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ER_LogArgTaylor(sM, dist) ));
			iLField[0] += ( m3[0] * tmpScalar );
			iLField[1] += ( m3[1] * tmpScalar );
			iLField[2] += ( m3[2] * tmpScalar );
			iLPhi += ( t3 * tmpScalar );
		}
	}
#endif /* KEMFIELD_CUFASTRWG */

#ifndef KEMFIELD_CUFASTRWG
	if( lineIndex != 3 ) {
#endif /* KEMFIELD_CUFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		tmpScalar = (LOG( logArgNom )-LOG( logArgDenom ));
		iLField[0] += ( m3[0] * tmpScalar );
		iLField[1] += ( m3[1] * tmpScalar );
		iLField[2] += ( m3[2] * tmpScalar );
		iLPhi += ( t3 * tmpScalar );
#ifndef KEMFIELD_CUFASTRWG
	}
#endif /* KEMFIELD_CUFASTRWG */

	return MAKECU4( iLField[0], iLField[1], iLField[2], iLPhi );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE ER_Potential( const CU_TYPE* P, const CU_TYPE* data )
{
#ifndef KEMFIELD_CUFASTRWG
    // corner points P0, P1, P2 and P3

    const CU_TYPE rectP0[3] = {
    		data[2], data[3], data[4] };

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
#endif /* KEMFIELD_CUFASTRWG */

	// get rectangle centroid

	CU_TYPE rectCenter[3];
	Rect_Centroid( rectCenter, data );

	// get perpendicular normal vector n3 on rectangle surface

	CU_TYPE rectN3[3];
	Rect_Normal( rectN3, data );

#ifndef KEMFIELD_CUFASTRWG
    // side line vectors

    const CU_TYPE rectAlongSideP0P1[3] = {
    		data[0]*data[5],
			data[0]*data[6],
			data[0]*data[7] }; // = A * N1
    const CU_TYPE rectAlongSideP1P2[3] = {
    		data[1]*data[8],
			data[1]*data[9],
			data[1]*data[10] }; // = B * N2
    const CU_TYPE rectAlongSideP2P3[3] = {
    		-data[0]*data[5],
			-data[0]*data[6],
			-data[0]*data[7] }; // = -A * N1
    const CU_TYPE rectAlongSideP3P0[3] = {
    		-data[1]*data[8],
			-data[1]*data[9],
			-data[1]*data[10] }; // = -B * N2

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const CU_TYPE rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const CU_TYPE rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const CU_TYPE rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const CU_TYPE rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };
#endif /* KEMFIELD_CUFASTRWG */

	// check if field point is at rectangle edge or at side line

    CU_TYPE distToLineMin = 0.;
    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on rectangle edge (= 2) or point is on side line (= 1)

#ifndef KEMFIELD_CUFASTRWG
    CU_TYPE distToLine = 0.;
    CU_TYPE lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    CU_TYPE tmpVector[3] = {0., 0., 0.};
    CU_TYPE tmpScalar = 0.;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./data[0];

    Compute_CrossProduct( rectAlongSideP0P1, rectDistP0, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0
    lineLambda = ((-rectDistP0[0]*rectAlongSideP0P1[0]) + (-rectDistP0[1]*rectAlongSideP0P1[1]) + (-rectDistP0[2]*rectAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./data[1];

    Compute_CrossProduct( rectAlongSideP1P2, rectDistP1, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP1[0]*rectAlongSideP1P2[0]) + (-rectDistP1[1]*rectAlongSideP1P2[1]) + (-rectDistP1[2]*rectAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P3 side line

    tmpScalar = 1./data[0];

    Compute_CrossProduct( rectAlongSideP2P3, rectDistP2, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP2[0]*rectAlongSideP2P3[0]) + (-rectDistP2[1]*rectAlongSideP2P3[1]) + (-rectDistP2[2]*rectAlongSideP2P3[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // 3 - check distances to P3P0 side line

    tmpScalar = 1./data[1];

    Compute_CrossProduct( rectAlongSideP3P0, rectDistP3, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for rectDistP3
    lineLambda = ((-rectDistP3[0]*rectAlongSideP3P0[0]) + (-rectDistP3[1]*rectAlongSideP3P0[1]) + (-rectDistP3[2]*rectAlongSideP3P0[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 3;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length 'CORRECTIONTRIN3' in positive and negative N3 direction

    if( correctionCounter == 2 ) {
    	const CU_TYPE upEps[3] = {
    			P[0] + CORRECTIONRECTN3*rectN3[0],
				P[1] + CORRECTIONRECTN3*rectN3[1],
				P[2] + CORRECTIONRECTN3*rectN3[2]
    	};
    	const CU_TYPE downEps[3] = {
    			P[0] - CORRECTIONRECTN3*rectN3[0],
				P[1] - CORRECTIONRECTN3*rectN3[1],
				P[2] - CORRECTIONRECTN3*rectN3[2]
    	};

    	// compute IqS term

        const CU_TYPE hUp = ( rectN3[0] * (upEps[0]-rectCenter[0]) )
				+ ( rectN3[1] * (upEps[1]-rectCenter[1]) )
				+ ( rectN3[2] * (upEps[2]-rectCenter[2]) );

        const CU_TYPE solidAngleUp = Rect_SolidAngle( upEps, data );

        const CU_TYPE hDown = ( rectN3[0] * (downEps[0]-rectCenter[0]) )
				+ ( rectN3[1] * (downEps[1]-rectCenter[1]) )
				+ ( rectN3[2] * (downEps[2]-rectCenter[2]) );

        const CU_TYPE solidAngleDown = Rect_SolidAngle( downEps, data );

    	// compute IqL

    	const CU_TYPE IqLUp = ER_IqLPotential( 9, 9, 9, upEps, data ); /* no line correction */

    	const CU_TYPE IqLDown = ER_IqLPotential( 9, 9, 9, downEps, data ); /* no line correction */

    	const CU_TYPE finalResult = 0.5*((-hUp*solidAngleUp - IqLUp) + (-hDown*solidAngleDown - IqLDown));

    	return finalResult*M_ONEOVER_4PI_EPS0;
    }
#endif /* KEMFIELD_CUFASTRWG */

    const CU_TYPE h = ( rectN3[0] * (P[0]-rectCenter[0]) )
			+ ( rectN3[1] * (P[1]-rectCenter[1]) )
			+ ( rectN3[2] * (P[2]-rectCenter[2]) );

    const CU_TYPE rectSolidAngle = Rect_SolidAngle( P, data );

    const CU_TYPE finalResult = (-h*rectSolidAngle) - ER_IqLPotential( correctionCounter, correctionLineIndex, distToLineMin, P, data );

	return finalResult*M_ONEOVER_4PI_EPS0;
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ER_EField( const CU_TYPE* P, const CU_TYPE* data )
{
#ifndef KEMFIELD_CUFASTRWG
    // corner points P0, P1, P2 and P3

    const CU_TYPE rectP0[3] = {
    		data[2], data[3], data[4] };

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
#endif /* KEMFIELD_CUFASTRWG */

	// get perpendicular normal vector n3 on rectangle surface

	CU_TYPE rectN3[3];
	Rect_Normal( rectN3, data );

#ifndef KEMFIELD_CUFASTRWG
    // side line vectors

    CU_TYPE rectAlongSideP0P1[3] = {
    		data[0]*data[5],
			data[0]*data[6],
			data[0]*data[7] }; // = A * N1
    CU_TYPE rectAlongSideP1P2[3] = {
    		data[1]*data[8],
			data[1]*data[9],
			data[1]*data[10] }; // = B * N2
    CU_TYPE rectAlongSideP2P3[3] = {
    		-data[0]*data[5],
			-data[0]*data[6],
			-data[0]*data[7] }; // = -A * N1
    CU_TYPE rectAlongSideP3P0[3] = {
    		-data[1]*data[8],
			-data[1]*data[9],
			-data[1]*data[10] }; // = -B * N2

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const CU_TYPE rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const CU_TYPE rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const CU_TYPE rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const CU_TYPE rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };
#endif /* KEMFIELD_CUFASTRWG */

	// check if field point is at rectangle edge or at side line

    CU_TYPE distToLineMin = 0.;
    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on rectangle edge (= 2) or point is on side line (= 1)

#ifndef KEMFIELD_CUFASTRWG
    CU_TYPE distToLine = 0.;
    CU_TYPE lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    CU_TYPE tmpVector[3] = {0., 0., 0.,};
    CU_TYPE tmpScalar = 0.;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./data[0];

    Compute_CrossProduct( rectAlongSideP0P1, rectDistP0, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0
    lineLambda = ((-rectDistP0[0]*rectAlongSideP0P1[0]) + (-rectDistP0[1]*rectAlongSideP0P1[1]) + (-rectDistP0[2]*rectAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./data[1];

    Compute_CrossProduct( rectAlongSideP1P2, rectDistP1, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP1[0]*rectAlongSideP1P2[0]) + (-rectDistP1[1]*rectAlongSideP1P2[1]) + (-rectDistP1[2]*rectAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P3 side line

    tmpScalar = 1./data[0];

    Compute_CrossProduct( rectAlongSideP2P3, rectDistP2, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP2[0]*rectAlongSideP2P3[0]) + (-rectDistP2[1]*rectAlongSideP2P3[1]) + (-rectDistP2[2]*rectAlongSideP2P3[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // 3 - check distances to P3P0 side line

    tmpScalar = 1./data[1];

    Compute_CrossProduct( rectAlongSideP3P0, rectDistP3, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for rectDistP3
    lineLambda = ((-rectDistP3[0]*rectAlongSideP3P0[0]) + (-rectDistP3[1]*rectAlongSideP3P0[1]) + (-rectDistP3[2]*rectAlongSideP3P0[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 3;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length 'CORRECTIONTRIN3' in positive and negative N3 direction

    if( correctionCounter == 2 ) {
    	const CU_TYPE upEps[3] = {
    			P[0] + CORRECTIONRECTN3*rectN3[0],
				P[1] + CORRECTIONRECTN3*rectN3[1],
				P[2] + CORRECTIONRECTN3*rectN3[2]
    	};
    	const CU_TYPE downEps[3] = {
    			P[0] - CORRECTIONRECTN3*rectN3[0],
				P[1] - CORRECTIONRECTN3*rectN3[1],
				P[2] - CORRECTIONRECTN3*rectN3[2]
    	};

    	// compute IqS term

        const CU_TYPE solidAngleUp = Rect_SolidAngle( upEps, data );
        const CU_TYPE solidAngleDown = Rect_SolidAngle( downEps, data );

        // compute IqL

    	const CU_TYPE4 IqLUp = ER_IqLField( 9, 9, 9, upEps, data ); /* no line correction */
    	const CU_TYPE4 IqLDown = ER_IqLField( 9, 9, 9, downEps, data ); /* no line correction */

    	const CU_TYPE finalResult[3] = {
    			0.5*M_ONEOVER_4PI_EPS0*((rectN3[0]*solidAngleUp + IqLUp.x) + (rectN3[0]*solidAngleDown + IqLDown.x)),
    			0.5*M_ONEOVER_4PI_EPS0*((rectN3[1]*solidAngleUp + IqLUp.y) + (rectN3[1]*solidAngleDown + IqLDown.y)),
    			0.5*M_ONEOVER_4PI_EPS0*((rectN3[2]*solidAngleUp + IqLUp.z) + (rectN3[2]*solidAngleDown + IqLDown.z)),
    	};

        return MAKECU4( finalResult[0], finalResult[1], finalResult[2], 0. );
    }
#endif /* KEMFIELD_CUFASTRWG */

    const CU_TYPE rectSolidAngle = Rect_SolidAngle( P, data );
	const CU_TYPE4 IqLField = ER_IqLField( correctionCounter, correctionLineIndex, distToLineMin, P, data );

	const CU_TYPE finalResult[3] = {
			M_ONEOVER_4PI_EPS0*(rectN3[0]*rectSolidAngle + IqLField.x),
			M_ONEOVER_4PI_EPS0*(rectN3[1]*rectSolidAngle + IqLField.y),
			M_ONEOVER_4PI_EPS0*(rectN3[2]*rectSolidAngle + IqLField.z)
	};

	return MAKECU4( finalResult[0], finalResult[1], finalResult[2], 0. );
}

//______________________________________________________________________________

__forceinline__ __device__
CU_TYPE4 ER_EFieldAndPotential( const CU_TYPE* P, const CU_TYPE* data )
{
#ifndef KEMFIELD_CUFASTRWG
    // corner points P0, P1, P2 and P3

    const CU_TYPE rectP0[3] = {
    		data[2], data[3], data[4] };

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
#endif /* KEMFIELD_CUFASTRWG */

	// get perpendicular normal vector n3 on rectangle surface

	CU_TYPE rectN3[3];
	Rect_Normal( rectN3, data );

	// get rectangle centroid

	CU_TYPE rectCenter[3];
	Rect_Centroid( rectCenter, data );

#ifndef KEMFIELD_CUFASTRWG
    // side line vectors

    CU_TYPE rectAlongSideP0P1[3] = {
    		data[0]*data[5],
			data[0]*data[6],
			data[0]*data[7] }; // = A * N1
    CU_TYPE rectAlongSideP1P2[3] = {
    		data[1]*data[8],
			data[1]*data[9],
			data[1]*data[10] }; // = B * N2
    CU_TYPE rectAlongSideP2P3[3] = {
    		-data[0]*data[5],
			-data[0]*data[6],
			-data[0]*data[7] }; // = -A * N1
    CU_TYPE rectAlongSideP3P0[3] = {
    		-data[1]*data[8],
			-data[1]*data[9],
			-data[1]*data[10] }; // = -B * N2

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const CU_TYPE rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const CU_TYPE rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const CU_TYPE rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const CU_TYPE rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };
#endif /* KEMFIELD_CUFASTRWG */

	// check if field point is at rectangle edge or at side line

    CU_TYPE distToLineMin = 0.;
    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on rectangle edge (= 2) or point is on side line (= 1)

#ifndef KEMFIELD_CUFASTRWG
    CU_TYPE distToLine = 0.;
    CU_TYPE lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    CU_TYPE tmpVector[3] = {0., 0., 0.};
    CU_TYPE tmpScalar = 0.;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./data[0];

    Compute_CrossProduct( rectAlongSideP0P1, rectDistP0, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0
    lineLambda = ((-rectDistP0[0]*rectAlongSideP0P1[0]) + (-rectDistP0[1]*rectAlongSideP0P1[1]) + (-rectDistP0[2]*rectAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./data[1];

    Compute_CrossProduct( rectAlongSideP1P2, rectDistP1, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP1[0]*rectAlongSideP1P2[0]) + (-rectDistP1[1]*rectAlongSideP1P2[1]) + (-rectDistP1[2]*rectAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P3 side line

    tmpScalar = 1./data[0];

    Compute_CrossProduct( rectAlongSideP2P3, rectDistP2, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP2[0]*rectAlongSideP2P3[0]) + (-rectDistP2[1]*rectAlongSideP2P3[1]) + (-rectDistP2[2]*rectAlongSideP2P3[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // 3 - check distances to P3P0 side line

    tmpScalar = 1./data[1];

    Compute_CrossProduct( rectAlongSideP3P0, rectDistP3, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for rectDistP3
    lineLambda = ((-rectDistP3[0]*rectAlongSideP3P0[0]) + (-rectDistP3[1]*rectAlongSideP3P0[1]) + (-rectDistP3[2]*rectAlongSideP3P0[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETORECTLINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 3;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length 'CORRECTIONTRIN3' in positive and negative N3 direction

    if( correctionCounter == 2 ) {
    	const CU_TYPE upEps[3] = {
    			P[0] + CORRECTIONRECTN3*rectN3[0],
				P[1] + CORRECTIONRECTN3*rectN3[1],
				P[2] + CORRECTIONRECTN3*rectN3[2]
    	};
    	const CU_TYPE downEps[3] = {
    			P[0] - CORRECTIONRECTN3*rectN3[0],
				P[1] - CORRECTIONRECTN3*rectN3[1],
				P[2] - CORRECTIONRECTN3*rectN3[2]
    	};

    	// compute IqS term

        const CU_TYPE hUp = ( rectN3[0] * (upEps[0]-rectCenter[0]) )
				+ ( rectN3[1] * (upEps[1]-rectCenter[1]) )
				+ ( rectN3[2] * (upEps[2]-rectCenter[2]) );

        const CU_TYPE solidAngleUp = Rect_SolidAngle( upEps, data );

        const CU_TYPE hDown = ( rectN3[0] * (downEps[0]-rectCenter[0]) )
				+ ( rectN3[1] * (downEps[1]-rectCenter[1]) )
				+ ( rectN3[2] * (downEps[2]-rectCenter[2]) );

        const CU_TYPE solidAngleDown = Rect_SolidAngle( downEps, data );

        // compute IqL

		const CU_TYPE4 IqLFieldAndPotUp = ER_IqLFieldAndPotential( 9, 9, 9, upEps, data );
		const CU_TYPE4 IqLFieldAndPotDown = ER_IqLFieldAndPotential( 9, 9, 9, downEps, data );

    	const CU_TYPE finalField[3] = {
    			0.5*M_ONEOVER_4PI_EPS0*((rectN3[0]*solidAngleUp + IqLFieldAndPotUp.x) + (rectN3[0]*solidAngleDown + IqLFieldAndPotDown.x)),
				0.5*M_ONEOVER_4PI_EPS0*((rectN3[1]*solidAngleUp + IqLFieldAndPotUp.y) + (rectN3[1]*solidAngleDown + IqLFieldAndPotDown.y)),
				0.5*M_ONEOVER_4PI_EPS0*((rectN3[2]*solidAngleUp + IqLFieldAndPotUp.z) + (rectN3[2]*solidAngleDown + IqLFieldAndPotDown.z)) };
    	const CU_TYPE finalPotential = 0.5*M_ONEOVER_4PI_EPS0*((-hUp*solidAngleUp -  IqLFieldAndPotUp.w) + (-hDown*solidAngleDown -  IqLFieldAndPotDown.w));

    	return MAKECU4( finalField[0], finalField[1], finalField[2], finalPotential );
    }
#endif /* KEMFIELD_CUFASTRWG */

    const CU_TYPE h = ( rectN3[0] * (P[0]-rectCenter[0]) )
			+ ( rectN3[1] * (P[1]-rectCenter[1]) )
			+ ( rectN3[2] * (P[2]-rectCenter[2]) );
    const CU_TYPE rectSolidAngle = Rect_SolidAngle( P, data );

	const CU_TYPE4 IqLFieldAndPotUp = ER_IqLFieldAndPotential( correctionCounter, correctionLineIndex, distToLineMin, P, data );

	const CU_TYPE finalField[3] = {
			M_ONEOVER_4PI_EPS0*(rectN3[0]*rectSolidAngle + IqLFieldAndPotUp.x),
			M_ONEOVER_4PI_EPS0*(rectN3[1]*rectSolidAngle + IqLFieldAndPotUp.y),
			M_ONEOVER_4PI_EPS0*(rectN3[2]*rectSolidAngle + IqLFieldAndPotUp.z) };
	const CU_TYPE finalPotential = M_ONEOVER_4PI_EPS0*(-h*rectSolidAngle -  IqLFieldAndPotUp.w);

	return MAKECU4( finalField[0], finalField[1], finalField[2], finalPotential );
}

//______________________________________________________________________________

#endif /* KEMFIELD_ELECTROSTATICRWGRECTANGLE_CUH */
