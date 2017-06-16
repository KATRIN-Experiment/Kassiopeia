#ifndef KEMFIELD_ELECTROSTATICRWGTRIANGLE_CL
#define KEMFIELD_ELECTROSTATICRWGTRIANGLE_CL

// OpenCL kernel for integration of triangle surface for electrostatics in RWG basis
// Detailed information on the implementation can be found in the CPU code,
// class 'KElectrostaticRWGTriangleIntegrator'.
// Author: Daniel Hilk
//
// Recommended workgroup sizes by driver for NVIDIA Tesla K40c,
// sizes for fast option given in brackets.
// * ET_Potential: 384 (640)
// * ET_EField: 512 (640)
// * ET_EFieldAndPotential: 384 (512)

#include "kEMField_Triangle.cl"
#include "kEMField_VectorOperations.cl"
#include "kEMField_SolidAngle.cl"

// Triangle geometry definition (as defined by the streamers in KTriangle.hh):
//
// data[0]:     A
// data[1]:     B
// data[2..4]:  P0[0..2]
// data[5..7]:  N1[0..2]
// data[8..10]: N2[0..2]

#define MINDISTANCETOTRILINE 1.E-14
#define CORRECTIONTRIN3 1.E-7 /* step in N3 direction if field point is on edge */
#define TRILOGARGQUOTIENT 1.E-6 /* limit for Taylor expansion if field point is on line (=dist/sM) */

//______________________________________________________________________________

#if KEMFIELD_OCLFASTRWG==0
CL_TYPE ET_LogArgTaylor( const CL_TYPE sMin, const CL_TYPE dist )
{
	CL_TYPE quotient = fabs(dist/sMin);
	if( quotient < 1.e-14 ) quotient = 1.e-14;
	return 0.5*fabs(sMin)*POW2(quotient);
}
#endif /* KEMFIELD_OCLFASTRWG */

//______________________________________________________________________________

CL_TYPE ET_IqLPotential( const unsigned short countCross, const unsigned short lineIndex,
		const CL_TYPE dist, const CL_TYPE* P, __global const CL_TYPE* data )
{
    // corner points P0, P1 and P2

    const CL_TYPE triP0[3] = {
    		data[2], data[3], data[4] };

    const CL_TYPE triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const CL_TYPE triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on triangle surface

    CL_TYPE triN3[3];
    Tri_Normal( triN3, data );

    // side line unit vectors

    CL_TYPE triAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    CL_TYPE triAlongSideP1P2Unit[3];
    Compute_UnitVector( triP1, triP2, triAlongSideP1P2Unit );
    CL_TYPE triAlongSideP2P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

	// length values of side lines, only half value is needed

    const CL_TYPE triAlongSideHalfLengthP0P1 = 0.5 * data[0];
    const CL_TYPE triAlongSideHalfLengthP1P2 = 0.5 * SQRT( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );
	const CL_TYPE triAlongSideHalfLengthP2P0 = 0.5 * data[1];

    // center point of each side to field point

    const CL_TYPE e0[3] = {
    		triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[0] + triP0[0],
			triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[1] + triP0[1],
			triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[2] + triP0[2] };

    const CL_TYPE e1[3] = {
    		triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[0] + triP1[0],
			triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[1] + triP1[1],
			triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[2] + triP1[2] };

    const CL_TYPE e2[3] = {
    		triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[0] + triP2[0],
			triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[1] + triP2[1],
			triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[2] + triP2[2] };

    // outward pointing vector m, perpendicular to side lines

    CL_TYPE m0[3];
    Compute_CrossProduct( triAlongSideP0P1Unit, triN3, m0 );
    CL_TYPE m1[3];
    Compute_CrossProduct( triAlongSideP1P2Unit, triN3, m1 );
    CL_TYPE m2[3];
    Compute_CrossProduct( triAlongSideP2P0Unit, triN3, m2 );

    // size t

    const CL_TYPE t0 = (m0[0]*(P[0]-e0[0])) + (m0[1]*(P[1]-e0[1])) + (m0[2]*(P[2]-e0[2]));
    const CL_TYPE t1 = (m1[0]*(P[0]-e1[0])) + (m1[1]*(P[1]-e1[1])) + (m1[2]*(P[2]-e1[2]));
    const CL_TYPE t2 = (m2[0]*(P[0]-e2[0])) + (m2[1]*(P[1]-e2[1])) + (m2[2]*(P[2]-e2[2]));

    // distance between triangle vertex points and field points in positive rotation order
    // pointing to the triangle vertex point

    const CL_TYPE triDistP0[3] = {
    		triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };

    const CL_TYPE triDistP1[3] = {
    		triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };

    const CL_TYPE triDistP2[3] = {
    		triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };

    const CL_TYPE triMagDistP0 = SQRT( POW2(triDistP0[0]) + POW2(triDistP0[1]) + POW2(triDistP0[2]) );
    const CL_TYPE triMagDistP1 = SQRT( POW2(triDistP1[0]) + POW2(triDistP1[1]) + POW2(triDistP1[2]) );
    const CL_TYPE triMagDistP2 = SQRT( POW2(triDistP2[0]) + POW2(triDistP2[1]) + POW2(triDistP2[2]) );

    // evaluation of line integral

	CL_TYPE logArgNom = 0.;
	CL_TYPE logArgDenom = 0.;
	CL_TYPE iL = 0.;

	// 0 //

	CL_TYPE rM = triMagDistP0;
	CL_TYPE rP = triMagDistP1;
	CL_TYPE sM = (triDistP0[0]*triAlongSideP0P1Unit[0]) + (triDistP0[1]*triAlongSideP0P1Unit[1]) + (triDistP0[2]*triAlongSideP0P1Unit[2]);
	CL_TYPE sP = (triDistP1[0]*triAlongSideP0P1Unit[0]) + (triDistP1[1]*triAlongSideP0P1Unit[1]) + (triDistP1[2]*triAlongSideP0P1Unit[2]);

#if KEMFIELD_OCLFASTRWG==0
	if ( (countCross==1) && (lineIndex==0) ) {
		if( fabs(dist/sM) < TRILOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			iL += ( t0 * (LOG( logArgNom )-LOG( ET_LogArgTaylor(sM, dist) )) );
		}
	}
#endif /* KEMFIELD_OCLFASTRWG */

#if KEMFIELD_OCLFASTRWG==0
	if( lineIndex != 0 ) {
#endif /* KEMFIELD_OCLFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		iL += ( t0 * (LOG(logArgNom)-LOG(logArgDenom)) );
#if KEMFIELD_OCLFASTRWG==0
	}
#endif /* KEMFIELD_OCLFASTRWG */

	// 1 //

	rM = triMagDistP1;
	rP = triMagDistP2;
	sM = (triDistP1[0]*triAlongSideP1P2Unit[0]) + (triDistP1[1]*triAlongSideP1P2Unit[1]) + (triDistP1[2]*triAlongSideP1P2Unit[2]);
	sP = (triDistP2[0]*triAlongSideP1P2Unit[0]) + (triDistP2[1]*triAlongSideP1P2Unit[1]) + (triDistP2[2]*triAlongSideP1P2Unit[2]);

#if KEMFIELD_OCLFASTRWG==0
	if ( (countCross==1) && (lineIndex==1) ) {
		if( fabs(dist/sM) < TRILOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			iL += ( t1 * (LOG( logArgNom )-LOG( ET_LogArgTaylor(sM, dist) )) );
		}
	}
#endif /* KEMFIELD_OCLFASTRWG */

#if KEMFIELD_OCLFASTRWG==0
	if( lineIndex != 1 ) {
#endif /* KEMFIELD_OCLFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		iL += ( t1 * (LOG(logArgNom)-LOG(logArgDenom)) );
#if KEMFIELD_OCLFASTRWG==0
	}
#endif /* KEMFIELD_OCLFASTRWG */

	// 2 //

	rM = triMagDistP2;
	rP = triMagDistP0;
	sM = (triDistP2[0]*triAlongSideP2P0Unit[0]) + (triDistP2[1]*triAlongSideP2P0Unit[1]) + (triDistP2[2]*triAlongSideP2P0Unit[2]);
	sP = (triDistP0[0]*triAlongSideP2P0Unit[0]) + (triDistP0[1]*triAlongSideP2P0Unit[1]) + (triDistP0[2]*triAlongSideP2P0Unit[2]);

#if KEMFIELD_OCLFASTRWG==0
	if ( (countCross==1) && (lineIndex==2) ) {
		if( fabs(dist/sM) < TRILOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			iL += ( t2 * (LOG( logArgNom )-LOG( ET_LogArgTaylor(sM, dist) )) );
		}
	}
#endif /* KEMFIELD_OCLFASTRWG */

#if KEMFIELD_OCLFASTRWG==0
	if( lineIndex != 2 ) {
#endif /* KEMFIELD_OCLFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		iL += ( t2 * (LOG(logArgNom)-LOG(logArgDenom)) );
#if KEMFIELD_OCLFASTRWG==0
	}
#endif /* KEMFIELD_OCLFASTRWG */

	return iL;
}


//______________________________________________________________________________


CL_TYPE4 ET_IqLField( const unsigned short countCross, const unsigned short lineIndex, const CL_TYPE dist, const CL_TYPE* P, __global const CL_TYPE* data )
{
    // corner points P0, P1, P2 and P3

    const CL_TYPE triP0[3] = {
    		data[2], data[3], data[4] };

    const CL_TYPE triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const CL_TYPE triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on triangle surface

    CL_TYPE triN3[3];
    Tri_Normal( triN3, data );

    // side line unit vectors

    CL_TYPE triAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    CL_TYPE triAlongSideP1P2Unit[3];
    Compute_UnitVector( triP1, triP2, triAlongSideP1P2Unit );
    CL_TYPE triAlongSideP2P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

    // outward pointing vector m, perpendicular to side lines

    CL_TYPE m0[3];
    Compute_CrossProduct( triAlongSideP0P1Unit, triN3, m0 );
    CL_TYPE m1[3];
    Compute_CrossProduct( triAlongSideP1P2Unit, triN3, m1 );
    CL_TYPE m2[3];
    Compute_CrossProduct( triAlongSideP2P0Unit, triN3, m2 );

    // distance between triangle vertex points and field points in positive rotation order
    // pointing to the triangle vertex point

    const CL_TYPE triDistP0[3] = {
    		triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };

    const CL_TYPE triDistP1[3] = {
    		triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };

    const CL_TYPE triDistP2[3] = {
    		triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };

    const CL_TYPE triMagDistP0 = SQRT( POW2(triDistP0[0]) + POW2(triDistP0[1]) + POW2(triDistP0[2]) );
    const CL_TYPE triMagDistP1 = SQRT( POW2(triDistP1[0]) + POW2(triDistP1[1]) + POW2(triDistP1[2]) );
    const CL_TYPE triMagDistP2 = SQRT( POW2(triDistP2[0]) + POW2(triDistP2[1]) + POW2(triDistP2[2]) );

    // evaluation of line integral

	CL_TYPE logArgNom = 0.;
	CL_TYPE logArgDenom = 0.;
	CL_TYPE iL[3] = {0., 0., 0.};
	CL_TYPE tmpScalar = 0.;

	// 0 //

	CL_TYPE rM = triMagDistP0;
	CL_TYPE rP = triMagDistP1;
	CL_TYPE sM = (triDistP0[0]*triAlongSideP0P1Unit[0]) + (triDistP0[1]*triAlongSideP0P1Unit[1]) + (triDistP0[2]*triAlongSideP0P1Unit[2]);
	CL_TYPE sP = (triDistP1[0]*triAlongSideP0P1Unit[0]) + (triDistP1[1]*triAlongSideP0P1Unit[1]) + (triDistP1[2]*triAlongSideP0P1Unit[2]);

#if KEMFIELD_OCLFASTRWG==0
	if ( (countCross==1) && (lineIndex==0) ) {
		if( fabs(dist/sM) < TRILOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = LOG( logArgNom ) - LOG( ET_LogArgTaylor(sM, dist) );
			iL[0] += ( m0[0] * tmpScalar );
			iL[1] += ( m0[1] * tmpScalar );
			iL[2] += ( m0[2] * tmpScalar );
		}
	}
#endif /* KEMFIELD_OCLFASTRWG */

#if KEMFIELD_OCLFASTRWG==0
	if( lineIndex != 0 ) {
#endif /* KEMFIELD_OCLFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		tmpScalar = (LOG(logArgNom)-LOG(logArgDenom));
		iL[0] += ( m0[0] * tmpScalar );
		iL[1] += ( m0[1] * tmpScalar );
		iL[2] += ( m0[2] * tmpScalar );
#if KEMFIELD_OCLFASTRWG==0
	}
#endif /* KEMFIELD_OCLFASTRWG */

	// 1 //

	rM = triMagDistP1;
	rP = triMagDistP2;
	sM = (triDistP1[0]*triAlongSideP1P2Unit[0]) + (triDistP1[1]*triAlongSideP1P2Unit[1]) + (triDistP1[2]*triAlongSideP1P2Unit[2]);
	sP = (triDistP2[0]*triAlongSideP1P2Unit[0]) + (triDistP2[1]*triAlongSideP1P2Unit[1]) + (triDistP2[2]*triAlongSideP1P2Unit[2]);

#if KEMFIELD_OCLFASTRWG==0
	if ( (countCross==1) && (lineIndex==1) ) {
		if( fabs(dist/sM) < TRILOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = LOG( logArgNom ) - LOG( ET_LogArgTaylor(sM, dist) );
			iL[0] += ( m1[0] * tmpScalar );
			iL[1] += ( m1[1] * tmpScalar );
			iL[2] += ( m1[2] * tmpScalar );
		}
	}
#endif /* KEMFIELD_OCLFASTRWG */

#if KEMFIELD_OCLFASTRWG==0
	if( lineIndex != 1 ) {
#endif /* KEMFIELD_OCLFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		tmpScalar = (LOG(logArgNom)-LOG(logArgDenom));
		iL[0] += ( m1[0] * tmpScalar );
		iL[1] += ( m1[1] * tmpScalar );
		iL[2] += ( m1[2] * tmpScalar );
#if KEMFIELD_OCLFASTRWG==0
	}
#endif /* KEMFIELD_OCLFASTRWG */

	// 2 //

	rM = triMagDistP2;
	rP = triMagDistP0;
	sM = (triDistP2[0]*triAlongSideP2P0Unit[0]) + (triDistP2[1]*triAlongSideP2P0Unit[1]) + (triDistP2[2]*triAlongSideP2P0Unit[2]);
	sP = (triDistP0[0]*triAlongSideP2P0Unit[0]) + (triDistP0[1]*triAlongSideP2P0Unit[1]) + (triDistP0[2]*triAlongSideP2P0Unit[2]);

#if KEMFIELD_OCLFASTRWG==0
	if ( (countCross==1) && (lineIndex==2) ) {
		if( fabs(dist/sM) < TRILOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = LOG( logArgNom ) - LOG( ET_LogArgTaylor(sM, dist) );
			iL[0] += ( m2[0] * tmpScalar );
			iL[1] += ( m2[1] * tmpScalar );
			iL[2] += ( m2[2] * tmpScalar );
		}
	}
#endif /* KEMFIELD_OCLFASTRWG */

#if KEMFIELD_OCLFASTRWG==0
	if( lineIndex != 2 ) {
#endif /* KEMFIELD_OCLFASTRWG */
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		tmpScalar = (LOG(logArgNom)-LOG(logArgDenom));
		iL[0] += ( m2[0] * tmpScalar );
		iL[1] += ( m2[1] * tmpScalar );
		iL[2] += ( m2[2] * tmpScalar );
#if KEMFIELD_OCLFASTRWG==0
	}
#endif /* KEMFIELD_OCLFASTRWG */

	return (CL_TYPE4)( iL[0], iL[1], iL[2], 0. );
}

//______________________________________________________________________________

CL_TYPE4 ET_IqLFieldAndPotential( const unsigned short countCross, const unsigned short lineIndex,
		const CL_TYPE dist, const CL_TYPE* P, __global const CL_TYPE* data )
{
    // corner points P0, P1 and P2

    const CL_TYPE triP0[3] = {
    		data[2], data[3], data[4] };

    const CL_TYPE triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const CL_TYPE triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on triangle surface

    CL_TYPE triN3[3];
    Tri_Normal( triN3, data );

    // side line unit vectors

    CL_TYPE triAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    CL_TYPE triAlongSideP1P2Unit[3];
    Compute_UnitVector( triP1, triP2, triAlongSideP1P2Unit );
    CL_TYPE triAlongSideP2P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

	// length values of side lines, only half value is needed

    const CL_TYPE triAlongSideHalfLengthP0P1 = 0.5 * data[0];
    const CL_TYPE triAlongSideHalfLengthP1P2 = 0.5 * SQRT( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );
	const CL_TYPE triAlongSideHalfLengthP2P0 = 0.5 * data[1];

    // center point of each side to field point

    const CL_TYPE e0[3] = {
    		triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[0] + triP0[0],
			triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[1] + triP0[1],
			triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[2] + triP0[2] };

    const CL_TYPE e1[3] = {
    		triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[0] + triP1[0],
			triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[1] + triP1[1],
			triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[2] + triP1[2] };

    const CL_TYPE e2[3] = {
    		triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[0] + triP2[0],
			triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[1] + triP2[1],
			triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[2] + triP2[2] };

    // outward pointing vector m, perpendicular to side lines

    CL_TYPE m0[3];
    Compute_CrossProduct( triAlongSideP0P1Unit, triN3, m0 );
    CL_TYPE m1[3];
    Compute_CrossProduct( triAlongSideP1P2Unit, triN3, m1 );
    CL_TYPE m2[3];
    Compute_CrossProduct( triAlongSideP2P0Unit, triN3, m2 );

    // size t

    const CL_TYPE t0 = (m0[0]*(P[0]-e0[0])) + (m0[1]*(P[1]-e0[1])) + (m0[2]*(P[2]-e0[2]));
    const CL_TYPE t1 = (m1[0]*(P[0]-e1[0])) + (m1[1]*(P[1]-e1[1])) + (m1[2]*(P[2]-e1[2]));
    const CL_TYPE t2 = (m2[0]*(P[0]-e2[0])) + (m2[1]*(P[1]-e2[1])) + (m2[2]*(P[2]-e2[2]));


    // distance between triangle vertex points and field points in positive rotation order
    // pointing to the triangle vertex point

    const CL_TYPE triDistP0[3] = {
    		triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };

    const CL_TYPE triDistP1[3] = {
    		triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };

    const CL_TYPE triDistP2[3] = {
    		triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };

    const CL_TYPE triMagDistP0 = SQRT( POW2(triDistP0[0]) + POW2(triDistP0[1]) + POW2(triDistP0[2]) );
    const CL_TYPE triMagDistP1 = SQRT( POW2(triDistP1[0]) + POW2(triDistP1[1]) + POW2(triDistP1[2]) );
    const CL_TYPE triMagDistP2 = SQRT( POW2(triDistP2[0]) + POW2(triDistP2[1]) + POW2(triDistP2[2]) );

    // evaluation of line integral

	CL_TYPE logArgNom = 0.;
	CL_TYPE logArgDenom = 0.;
	CL_TYPE iLPhi = 0.;
	CL_TYPE iLField[3] = {0., 0., 0.};
	CL_TYPE tmpScalar = 0.;

	// 0 //

	CL_TYPE rM = triMagDistP0;
	CL_TYPE rP = triMagDistP1;
	CL_TYPE sM = (triDistP0[0]*triAlongSideP0P1Unit[0]) + (triDistP0[1]*triAlongSideP0P1Unit[1]) + (triDistP0[2]*triAlongSideP0P1Unit[2]);
	CL_TYPE sP = (triDistP1[0]*triAlongSideP0P1Unit[0]) + (triDistP1[1]*triAlongSideP0P1Unit[1]) + (triDistP1[2]*triAlongSideP0P1Unit[2]);

#if KEMFIELD_OCLFASTRWG==0
	if ( (countCross==1) && (lineIndex==0) ) {
		if( fabs(dist/sM) < TRILOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ET_LogArgTaylor(sM, dist) ));
			iLField[0] += ( m0[0] * tmpScalar );
			iLField[1] += ( m0[1] * tmpScalar );
			iLField[2] += ( m0[2] * tmpScalar );
			iLPhi += ( t0 * tmpScalar );
		}
	}
#endif /* KEMFIELD_OCLFASTRWG */

#if KEMFIELD_OCLFASTRWG==0
	if( lineIndex != 0 ) {
#endif /* KEMFIELD_OCLFASTRWG */
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
#if KEMFIELD_OCLFASTRWG==0
	}
#endif /* KEMFIELD_OCLFASTRWG */

	// 1 //

	rM = triMagDistP1;
	rP = triMagDistP2;
	sM = (triDistP1[0]*triAlongSideP1P2Unit[0]) + (triDistP1[1]*triAlongSideP1P2Unit[1]) + (triDistP1[2]*triAlongSideP1P2Unit[2]);
	sP = (triDistP2[0]*triAlongSideP1P2Unit[0]) + (triDistP2[1]*triAlongSideP1P2Unit[1]) + (triDistP2[2]*triAlongSideP1P2Unit[2]);

#if KEMFIELD_OCLFASTRWG==0
	if ( (countCross==1) && (lineIndex==1) ) {
		if( fabs(dist/sM) < TRILOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ET_LogArgTaylor(sM, dist) ));
			iLField[0] += ( m1[0] * tmpScalar );
			iLField[1] += ( m1[1] * tmpScalar );
			iLField[2] += ( m1[2] * tmpScalar );
			iLPhi += ( t1 * tmpScalar );
		}
	}
#endif /* KEMFIELD_OCLFASTRWG */

#if KEMFIELD_OCLFASTRWG==0
	if( lineIndex != 1 ) {
#endif /* KEMFIELD_OCLFASTRWG */
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
#if KEMFIELD_OCLFASTRWG==0
	}
#endif /* KEMFIELD_OCLFASTRWG */

	// 2 //

	rM = triMagDistP2;
	rP = triMagDistP0;
	sM = (triDistP2[0]*triAlongSideP2P0Unit[0]) + (triDistP2[1]*triAlongSideP2P0Unit[1]) + (triDistP2[2]*triAlongSideP2P0Unit[2]);
	sP = (triDistP0[0]*triAlongSideP2P0Unit[0]) + (triDistP0[1]*triAlongSideP2P0Unit[1]) + (triDistP0[2]*triAlongSideP2P0Unit[2]);

#if KEMFIELD_OCLFASTRWG==0
	if ( (countCross==1) && (lineIndex==2) ) {
		if( fabs(dist/sM) < TRILOGARGQUOTIENT ) {
			logArgNom = (rP+sP);
			tmpScalar = (LOG( logArgNom )-LOG( ET_LogArgTaylor(sM, dist) ));
			iLField[0] += ( m2[0] * tmpScalar );
			iLField[1] += ( m2[1] * tmpScalar );
			iLField[2] += ( m2[2] * tmpScalar );
			iLPhi += ( t2 * tmpScalar );
		}
	}
#endif /* KEMFIELD_OCLFASTRWG */

#if KEMFIELD_OCLFASTRWG==0
	if( lineIndex != 2 ) {
#endif /* KEMFIELD_OCLFASTRWG */
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
#if KEMFIELD_OCLFASTRWG==0
	}
#endif /* KEMFIELD_OCLFASTRWG */

	return (CL_TYPE4)( iLField[0], iLField[1], iLField[2], iLPhi );
}

//______________________________________________________________________________

CL_TYPE ET_Potential( const CL_TYPE* P, __global const CL_TYPE* data )
{
#if KEMFIELD_OCLFASTRWG==0
    // corner points P0, P1 and P2

    const CL_TYPE triP0[3] = {
    		data[2], data[3], data[4] };

    const CL_TYPE triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const CL_TYPE triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB
#endif /* KEMFIELD_OCLFASTRWG */

	// get perpendicular normal vector n3 on triangle surface

    CL_TYPE triN3[3];
    Tri_Normal( triN3, data );

    // triangle centroid

    CL_TYPE triCenter[3];
    Tri_Centroid( triCenter, data );

#if KEMFIELD_OCLFASTRWG==0
    // side line vectors

    const CL_TYPE triAlongSideP0P1[3] = {
    		data[0]*data[5],
			data[0]*data[6],
			data[0]*data[7] }; // = A * N1

    const CL_TYPE triAlongSideP1P2[3] = {
    		triP2[0]-triP1[0],
			triP2[1]-triP1[1],
			triP2[2]-triP1[2] };

    const CL_TYPE triAlongSideP2P0[3] = {
    		(-1)*data[1]*data[8],
			(-1)*data[1]*data[9],
			(-1)*data[1]*data[10] }; // = -B * N2

    // length values of side lines

    const CL_TYPE triAlongSideLengthP0P1 = data[0];
    const CL_TYPE triAlongSideLengthP1P2 = SQRT( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );
	const CL_TYPE triAlongSideLengthP2P0 = data[1];

	// distance between triangle vertex points and field points in positive rotation order
	// pointing to the triangle vertex point

	const CL_TYPE triDistP0[3] = {
			triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };
	const CL_TYPE triDistP1[3] = {
			triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };
	const CL_TYPE triDistP2[3] = {
			triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };
#endif /* KEMFIELD_OCLFASTRWG */

	// check if field point is at triangle edge or at side line

    CL_TYPE distToLineMin = 0.;
    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on triangle edge (= 2) or point is on side line (= 1)

#if KEMFIELD_OCLFASTRWG==0
    CL_TYPE distToLine = 0.;
    CL_TYPE lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    CL_TYPE tmpVector[3] = {0., 0., 0.};
    CL_TYPE tmpScalar = 0.;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./triAlongSideLengthP0P1;

    Compute_CrossProduct( triAlongSideP0P1, triDistP0, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0
    lineLambda = ((-triDistP0[0]*triAlongSideP0P1[0]) + (-triDistP0[1]*triAlongSideP0P1[1]) + (-triDistP0[2]*triAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETOTRILINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./triAlongSideLengthP1P2;

    Compute_CrossProduct( triAlongSideP1P2, triDistP1, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction triDistP1 vector
    lineLambda = ((-triDistP1[0]*triAlongSideP1P2[0]) + (-triDistP1[1]*triAlongSideP1P2[1]) + (-triDistP1[2]*triAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETOTRILINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P0 side line

    tmpScalar = 1./triAlongSideLengthP2P0;

    Compute_CrossProduct( triAlongSideP2P0, triDistP2, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for triDistP2
    lineLambda = ((-triDistP2[0]*triAlongSideP2P0[0]) + (-triDistP2[1]*triAlongSideP2P0[1]) + (-triDistP2[2]*triAlongSideP2P0[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETOTRILINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length 'CORRECTIONTRIN3' in positive and negative N3 direction

    if( correctionCounter == 2 ) {
    	const CL_TYPE upEps[3] = {
    			P[0] + CORRECTIONTRIN3*triN3[0],
				P[1] + CORRECTIONTRIN3*triN3[1],
				P[2] + CORRECTIONTRIN3*triN3[2]
    	};
    	const CL_TYPE downEps[3] = {
    			P[0] - CORRECTIONTRIN3*triN3[0],
				P[1] - CORRECTIONTRIN3*triN3[1],
				P[2] - CORRECTIONTRIN3*triN3[2]
    	};

    	// compute IqS term

        const CL_TYPE hUp = ( triN3[0] * (upEps[0]-triCenter[0]) )
				+ ( triN3[1] * (upEps[1]-triCenter[1]) )
				+ ( triN3[2] * (upEps[2]-triCenter[2]) );

        const CL_TYPE solidAngleUp = Tri_SolidAngle( upEps, data );

        const CL_TYPE hDown = ( triN3[0] * (downEps[0]-triCenter[0]) )
				+ ( triN3[1] * (downEps[1]-triCenter[1]) )
				+ ( triN3[2] * (downEps[2]-triCenter[2]) );

        const CL_TYPE solidAngleDown = Tri_SolidAngle( downEps, data );

    	// compute IqL

    	const CL_TYPE IqLUp = ET_IqLPotential( 9, 9, 9, upEps, data ); /* no line correction */

    	const CL_TYPE IqLDown = ET_IqLPotential( 9, 9, 9, downEps, data ); /* no line correction */

    	const CL_TYPE finalResult = 0.5*((-hUp*solidAngleUp - IqLUp) + (-hDown*solidAngleDown - IqLDown));

    	return finalResult*M_ONEOVER_4PI_EPS0;
    }
#endif /* KEMFIELD_OCLFASTRWG */

    const CL_TYPE h = ( triN3[0] * (P[0]-triCenter[0]) )
			+ ( triN3[1] * (P[1]-triCenter[1]) )
			+ ( triN3[2] * (P[2]-triCenter[2]) );

    const CL_TYPE triSolidAngle = Tri_SolidAngle( P, data );

	const CL_TYPE finalResult = (-h*triSolidAngle) - ET_IqLPotential( correctionCounter, correctionLineIndex, distToLineMin, P, data );

	return finalResult*M_ONEOVER_4PI_EPS0;
}

//______________________________________________________________________________

CL_TYPE4 ET_EField( const CL_TYPE* P, __global const CL_TYPE* data )
{
#if KEMFIELD_OCLFASTRWG==0
    // corner points P0, P1 and P2

    const CL_TYPE triP0[3] = {
    		data[2], data[3], data[4] };

    const CL_TYPE triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const CL_TYPE triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB
#endif /* KEMFIELD_OCLFASTRWG */

	// get perpendicular normal vector n3 on triangle surface

    CL_TYPE triN3[3];
    Tri_Normal( triN3, data );

#if KEMFIELD_OCLFASTRWG==0
    // side line vectors

    const CL_TYPE triAlongSideP0P1[3] = {
    		data[0]*data[5],
			data[0]*data[6],
			data[0]*data[7] }; // = A * N1

    const CL_TYPE triAlongSideP1P2[3] = {
    		triP2[0]-triP1[0],
			triP2[1]-triP1[1],
			triP2[2]-triP1[2] };

    const CL_TYPE triAlongSideP2P0[3] = {
    		(-1)*data[1]*data[8],
			(-1)*data[1]*data[9],
			(-1)*data[1]*data[10] }; // = -B * N2

    // length values of side lines

    const CL_TYPE triAlongSideLengthP0P1 = data[0];
    const CL_TYPE triAlongSideLengthP1P2 = SQRT( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );
	const CL_TYPE triAlongSideLengthP2P0 = data[1];

	// distance between triangle vertex points and field points in positive rotation order
	// pointing to the triangle vertex point

	const CL_TYPE triDistP0[3] = {
			triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };
	const CL_TYPE triDistP1[3] = {
			triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };
	const CL_TYPE triDistP2[3] = {
			triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };
#endif /* KEMFIELD_OCLFASTRWG */

	// check if field point is at triangle edge or at side line

    CL_TYPE distToLineMin = 0.;
    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on triangle edge (= 2) or point is on side line (= 1)

#if KEMFIELD_OCLFASTRWG==0
    CL_TYPE distToLine = 0.;
    CL_TYPE lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    CL_TYPE tmpVector[3] = {0., 0., 0.};
    CL_TYPE tmpScalar = 0.;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./triAlongSideLengthP0P1;

    Compute_CrossProduct( triAlongSideP0P1, triDistP0, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0 in array
    lineLambda = ((-triDistP0[0]*triAlongSideP0P1[0]) + (-triDistP0[1]*triAlongSideP0P1[1]) + (-triDistP0[2]*triAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETOTRILINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./triAlongSideLengthP1P2;

    Compute_CrossProduct( triAlongSideP1P2, triDistP1, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction triDistP1 vector
    lineLambda = ((-triDistP1[0]*triAlongSideP1P2[0]) + (-triDistP1[1]*triAlongSideP1P2[1]) + (-triDistP1[2]*triAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETOTRILINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P0 side line

    tmpScalar = 1./triAlongSideLengthP2P0;

    Compute_CrossProduct( triAlongSideP2P0, triDistP2, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for triDistP2
    lineLambda = ((-triDistP2[0]*triAlongSideP2P0[0]) + (-triDistP2[1]*triAlongSideP2P0[1]) + (-triDistP2[2]*triAlongSideP2P0[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETOTRILINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length 'CORRECTIONTRIN3' in positive and negative N3 direction

    if( correctionCounter == 2 ) {
    	const CL_TYPE upEps[3] = {
    			P[0] + CORRECTIONTRIN3*triN3[0],
				P[1] + CORRECTIONTRIN3*triN3[1],
				P[2] + CORRECTIONTRIN3*triN3[2]
    	};
    	const CL_TYPE downEps[3] = {
    			P[0] - CORRECTIONTRIN3*triN3[0],
				P[1] - CORRECTIONTRIN3*triN3[1],
				P[2] - CORRECTIONTRIN3*triN3[2]
    	};

    	// compute IqS term

        const CL_TYPE solidAngleUp = Tri_SolidAngle( upEps, data );
        const CL_TYPE solidAngleDown = Tri_SolidAngle( downEps, data );

    	// compute IqL

    	const CL_TYPE4 IqLUp = ET_IqLField( 9, 9, 9, upEps, data ); /* no line correction */
    	const CL_TYPE4 IqLDown = ET_IqLField( 9, 9, 9, downEps, data ); /* no line correction */

    	const CL_TYPE finalResult[3] = {
    			0.5*M_ONEOVER_4PI_EPS0*((triN3[0]*solidAngleUp + IqLUp.s0) + (triN3[0]*solidAngleDown + IqLDown.s0)),
				0.5*M_ONEOVER_4PI_EPS0*((triN3[1]*solidAngleUp + IqLUp.s1) + (triN3[1]*solidAngleDown + IqLDown.s1)),
				0.5*M_ONEOVER_4PI_EPS0*((triN3[2]*solidAngleUp + IqLUp.s2) + (triN3[2]*solidAngleDown + IqLDown.s2))
    	};

    	return (CL_TYPE4)( finalResult[0], finalResult[1], finalResult[2], 0. );
    }
#endif /* KEMFIELD_OCLFASTRWG */
    const CL_TYPE triSolidAngle = Tri_SolidAngle( P, data );
	const CL_TYPE4 IqLField = ET_IqLField( correctionCounter, correctionLineIndex, distToLineMin, P, data );

	const CL_TYPE finalResult[3] = {
			M_ONEOVER_4PI_EPS0*(triN3[0]*triSolidAngle + IqLField.s0),
			M_ONEOVER_4PI_EPS0*(triN3[1]*triSolidAngle + IqLField.s1),
			M_ONEOVER_4PI_EPS0*(triN3[2]*triSolidAngle + IqLField.s2)
	};

    return (CL_TYPE4)( finalResult[0], finalResult[1], finalResult[2], 0. );
}

//______________________________________________________________________________

CL_TYPE4 ET_EFieldAndPotential( const CL_TYPE* P, __global const CL_TYPE* data )
{
#if KEMFIELD_OCLFASTRWG==0
    // corner points P0, P1 and P2

    const CL_TYPE triP0[3] = {
    		data[2], data[3], data[4] };

    const CL_TYPE triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const CL_TYPE triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB
#endif /* KEMFIELD_OCLFASTRWG */

	// get perpendicular normal vector n3 on triangle surface

    CL_TYPE triN3[3];
    Tri_Normal( triN3, data );

    // triangle centroid

    CL_TYPE triCenter[3];
    Tri_Centroid( triCenter, data );

#if KEMFIELD_OCLFASTRWG==0
    // side line vectors

    const CL_TYPE triAlongSideP0P1[3] = {
    		data[0]*data[5],
			data[0]*data[6],
			data[0]*data[7] }; // = A * N1

    const CL_TYPE triAlongSideP1P2[3] = {
    		triP2[0]-triP1[0],
			triP2[1]-triP1[1],
			triP2[2]-triP1[2] };

    const CL_TYPE triAlongSideP2P0[3] = {
    		(-1)*data[1]*data[8],
			(-1)*data[1]*data[9],
			(-1)*data[1]*data[10] }; // = -B * N2

    // length values of side lines

    const CL_TYPE triAlongSideLengthP0P1 = data[0];
    const CL_TYPE triAlongSideLengthP1P2 = SQRT( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );
	const CL_TYPE triAlongSideLengthP2P0 = data[1];

	// distance between triangle vertex points and field points in positive rotation order
	// pointing to the triangle vertex point

	const CL_TYPE triDistP0[3] = {
			triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };
	const CL_TYPE triDistP1[3] = {
			triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };
	const CL_TYPE triDistP2[3] = {
			triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };
#endif /* KEMFIELD_OCLFASTRWG */

	// check if field point is at triangle edge or at side line

    CL_TYPE distToLineMin = 0.;
    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on triangle edge (= 2) or point is on side line (= 1)

#if KEMFIELD_OCLFASTRWG==0
    CL_TYPE distToLine = 0.;
    CL_TYPE lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    CL_TYPE tmpVector[3] = {0., 0., 0.};
    CL_TYPE tmpScalar = 0.;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./triAlongSideLengthP0P1;

    Compute_CrossProduct( triAlongSideP0P1, triDistP0, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0
    lineLambda = ((-triDistP0[0]*triAlongSideP0P1[0]) + (-triDistP0[1]*triAlongSideP0P1[1]) + (-triDistP0[2]*triAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETOTRILINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./triAlongSideLengthP1P2;

    Compute_CrossProduct( triAlongSideP1P2, triDistP1, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction triDistP1 vector
    lineLambda = ((-triDistP1[0]*triAlongSideP1P2[0]) + (-triDistP1[1]*triAlongSideP1P2[1]) + (-triDistP1[2]*triAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETOTRILINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P0 side line

    tmpScalar = 1./triAlongSideLengthP2P0;

    Compute_CrossProduct( triAlongSideP2P0, triDistP2, tmpVector );

    distToLine = SQRT( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for triDistP2
    lineLambda = ((-triDistP2[0]*triAlongSideP2P0[0]) + (-triDistP2[1]*triAlongSideP2P0[1]) + (-triDistP2[2]*triAlongSideP2P0[2])) * POW2( tmpScalar );

    if( distToLine < MINDISTANCETOTRILINE ) {
        if( lineLambda>=-1.E-15 && lineLambda <=(1.+1.E-15) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length 'CORRECTIONTRIN3' in positive and negative N3 direction

    if( correctionCounter == 2 ) {
    	const CL_TYPE upEps[3] = {
    			P[0] + CORRECTIONTRIN3*triN3[0],
				P[1] + CORRECTIONTRIN3*triN3[1],
				P[2] + CORRECTIONTRIN3*triN3[2]
    	};
    	const CL_TYPE downEps[3] = {
    			P[0] - CORRECTIONTRIN3*triN3[0],
				P[1] - CORRECTIONTRIN3*triN3[1],
				P[2] - CORRECTIONTRIN3*triN3[2]
    	};

    	// compute IqS term

        const CL_TYPE hUp = ( triN3[0] * (upEps[0]-triCenter[0]) )
				+ ( triN3[1] * (upEps[1]-triCenter[1]) )
				+ ( triN3[2] * (upEps[2]-triCenter[2]) );

        const CL_TYPE solidAngleUp = Tri_SolidAngle( upEps, data );

        const CL_TYPE hDown = ( triN3[0] * (downEps[0]-triCenter[0]) )
				+ ( triN3[1] * (downEps[1]-triCenter[1]) )
				+ ( triN3[2] * (downEps[2]-triCenter[2]) );

        const CL_TYPE solidAngleDown = Tri_SolidAngle( downEps, data );

    	// compute IqL

		const CL_TYPE4 IqLFieldAndPotUp = ET_IqLFieldAndPotential( 9, 9, 9, upEps, data );
		const CL_TYPE4 IqLFieldAndPotDown = ET_IqLFieldAndPotential( 9, 9, 9, downEps, data );

    	const CL_TYPE finalField[3] = {
    			0.5*M_ONEOVER_4PI_EPS0*((triN3[0]*solidAngleUp + IqLFieldAndPotUp.s0) + (triN3[0]*solidAngleDown + IqLFieldAndPotDown.s0)),
				0.5*M_ONEOVER_4PI_EPS0*((triN3[1]*solidAngleUp + IqLFieldAndPotUp.s1) + (triN3[1]*solidAngleDown + IqLFieldAndPotDown.s1)),
				0.5*M_ONEOVER_4PI_EPS0*((triN3[2]*solidAngleUp + IqLFieldAndPotUp.s2) + (triN3[2]*solidAngleDown + IqLFieldAndPotDown.s2)) };
    	const CL_TYPE finalPotential = 0.5*M_ONEOVER_4PI_EPS0*((-hUp*solidAngleUp -  IqLFieldAndPotUp.s3) + (-hDown*solidAngleDown -  IqLFieldAndPotDown.s3));

    	return (CL_TYPE4)( finalField[0], finalField[1], finalField[2], finalPotential );
    }
#endif /* KEMFIELD_OCLFASTRWG */

    const CL_TYPE h = ( triN3[0] * (P[0]-triCenter[0]) )
			+ ( triN3[1] * (P[1]-triCenter[1]) )
			+ ( triN3[2] * (P[2]-triCenter[2]) );
    const CL_TYPE triSolidAngle = Tri_SolidAngle( P, data );

	const CL_TYPE4 IqLFieldAndPotUp = ET_IqLFieldAndPotential( correctionCounter, correctionLineIndex, distToLineMin, P, data );

	const CL_TYPE finalField[3] = {
			M_ONEOVER_4PI_EPS0*(triN3[0]*triSolidAngle + IqLFieldAndPotUp.s0),
			M_ONEOVER_4PI_EPS0*(triN3[1]*triSolidAngle + IqLFieldAndPotUp.s1),
			M_ONEOVER_4PI_EPS0*(triN3[2]*triSolidAngle + IqLFieldAndPotUp.s2) };
	const CL_TYPE finalPotential = M_ONEOVER_4PI_EPS0*(-h*triSolidAngle -  IqLFieldAndPotUp.s3);


	return (CL_TYPE4)( finalField[0], finalField[1], finalField[2], finalPotential );
}

//______________________________________________________________________________


#endif /* KEMFIELD_ELECTROSTATICRWGTRIANGLE_CL */
