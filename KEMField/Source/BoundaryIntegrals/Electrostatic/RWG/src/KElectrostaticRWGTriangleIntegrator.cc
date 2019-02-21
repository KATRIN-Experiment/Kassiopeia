#include "KElectrostaticRWGTriangleIntegrator.hh"

#define POW2(x) ((x)*(x))

namespace KEMField
{

double KElectrostaticRWGTriangleIntegrator::LogArgTaylor( const double sMin, const double dist ) const
{
	double quotient = fabs(dist/sMin);
	if( quotient < 1.e-14 ) quotient = 1.e-14;

	// Taylor expansion of log argument to second order
	double res = 0.5*fabs(sMin)*POW2(quotient);

	return res;
}

double KElectrostaticRWGTriangleIntegrator::IqLPotential( const double* data, const double* P,
		const unsigned short countCross, const unsigned short lineIndex, const double dist ) const
{
	// function computes second term of equation (63) in the case q = -1

    // corner points P0, P1 and P2

    const double triP0[3] = { data[2], data[3], data[4] };

    const double triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const double triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on triangle surface

    double triN3[3];
    triN3[0] = data[6]*data[10] - data[7]*data[9];
    triN3[1] = data[7]*data[8]  - data[5]*data[10];
    triN3[2] = data[5]*data[9]  - data[6]*data[8];
    const double triMagN3 = 1./sqrt( POW2(triN3[0])+POW2(triN3[1])+POW2(triN3[2]) );
    triN3[0] = triN3[0]*triMagN3;
    triN3[1] = triN3[1]*triMagN3;
    triN3[2] = triN3[2]*triMagN3;

    // side line unit vectors

    double triAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    double triAlongSideP1P2Unit[3];

	const double magP1P2 = 1./sqrt( POW2(triP2[0]-triP1[0])
			+ POW2(triP2[1]-triP1[1])
			+ POW2(triP2[2]-triP1[2]) );
	triAlongSideP1P2Unit[0] = magP1P2 * (triP2[0]-triP1[0]);
	triAlongSideP1P2Unit[1] = magP1P2 * (triP2[1]-triP1[1]);
	triAlongSideP1P2Unit[2] = magP1P2 * (triP2[2]-triP1[2]);

    double triAlongSideP2P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

	// length values of side lines, only half value is needed

    const double triAlongSideHalfLengthP0P1 = 0.5 * data[0];
    const double triAlongSideHalfLengthP1P2 = 0.5 * sqrt( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );
	const double triAlongSideHalfLengthP2P0 = 0.5 * data[1];

    // center point of each side

    const double e0[3] = {
    		triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[0] + triP0[0],
			triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[1] + triP0[1],
			triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[2] + triP0[2] };

    const double e1[3] = {
    		triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[0] + triP1[0],
			triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[1] + triP1[1],
			triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[2] + triP1[2] };

    const double e2[3] = {
    		triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[0] + triP2[0],
			triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[1] + triP2[1],
			triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[2] + triP2[2] };

    // outward pointing vector m, perpendicular to side lines

    const double m0[3] = {
    		(triAlongSideP0P1Unit[1]*triN3[2]) - (triAlongSideP0P1Unit[2]*triN3[1]),
			(triAlongSideP0P1Unit[2]*triN3[0]) - (triAlongSideP0P1Unit[0]*triN3[2]),
			(triAlongSideP0P1Unit[0]*triN3[1]) - (triAlongSideP0P1Unit[1]*triN3[0])
    };

    const double m1[3] = {
    		(triAlongSideP1P2Unit[1]*triN3[2]) - (triAlongSideP1P2Unit[2]*triN3[1]),
			(triAlongSideP1P2Unit[2]*triN3[0]) - (triAlongSideP1P2Unit[0]*triN3[2]),
			(triAlongSideP1P2Unit[0]*triN3[1]) - (triAlongSideP1P2Unit[1]*triN3[0])
    };

    const double m2[3] = {
    		(triAlongSideP2P0Unit[1]*triN3[2]) - (triAlongSideP2P0Unit[2]*triN3[1]),
			(triAlongSideP2P0Unit[2]*triN3[0]) - (triAlongSideP2P0Unit[0]*triN3[2]),
			(triAlongSideP2P0Unit[0]*triN3[1]) - (triAlongSideP2P0Unit[1]*triN3[0])
    };

    // size t

    const double t0 = (m0[0]*(P[0]-e0[0])) + (m0[1]*(P[1]-e0[1])) + (m0[2]*(P[2]-e0[2]));
    const double t1 = (m1[0]*(P[0]-e1[0])) + (m1[1]*(P[1]-e1[1])) + (m1[2]*(P[2]-e1[2]));
    const double t2 = (m2[0]*(P[0]-e2[0])) + (m2[1]*(P[1]-e2[1])) + (m2[2]*(P[2]-e2[2]));

    // distance between triangle vertex points and field points in positive rotation order
    // pointing to the triangle vertex point

    const double triDistP0[3] = {
    		triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };

    const double triDistP1[3] = {
    		triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };

    const double triDistP2[3] = {
    		triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };

    const double triMagDistP0 = sqrt( POW2(triDistP0[0]) + POW2(triDistP0[1]) + POW2(triDistP0[2]) );
    const double triMagDistP1 = sqrt( POW2(triDistP1[0]) + POW2(triDistP1[1]) + POW2(triDistP1[2]) );
    const double triMagDistP2 = sqrt( POW2(triDistP2[0]) + POW2(triDistP2[1]) + POW2(triDistP2[2]) );

    // evaluation of line integral

    double logArgNom, logArgDenom;
    double iL = 0.;

	// 0 //

    double rM = triMagDistP0;
    double rP = triMagDistP1;
    double sM = (triDistP0[0]*triAlongSideP0P1Unit[0]) + (triDistP0[1]*triAlongSideP0P1Unit[1]) + (triDistP0[2]*triAlongSideP0P1Unit[2]);
    double sP = (triDistP1[0]*triAlongSideP0P1Unit[0]) + (triDistP1[1]*triAlongSideP0P1Unit[1]) + (triDistP1[2]*triAlongSideP0P1Unit[2]);

	if ( (countCross==1) && (lineIndex==0) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			iL += ( t0 * (log( logArgNom )-log( LogArgTaylor(sM, dist) )) );
		}
	}

	if( lineIndex != 0 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		iL += ( t0 * (log(logArgNom)-log(logArgDenom)) );
	}

	// 1 //

	rM = triMagDistP1;
	rP = triMagDistP2;
	sM = (triDistP1[0]*triAlongSideP1P2Unit[0]) + (triDistP1[1]*triAlongSideP1P2Unit[1]) + (triDistP1[2]*triAlongSideP1P2Unit[2]);
	sP = (triDistP2[0]*triAlongSideP1P2Unit[0]) + (triDistP2[1]*triAlongSideP1P2Unit[1]) + (triDistP2[2]*triAlongSideP1P2Unit[2]);

	if ( (countCross==1) && (lineIndex==1) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			iL += ( t1 * (log( logArgNom )-log( LogArgTaylor(sM, dist) )) );
		}
	}

	if( lineIndex != 1 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		iL += ( t1 * (log(logArgNom)-log(logArgDenom)) );
	}

	// 2 //

	rM = triMagDistP2;
	rP = triMagDistP0;
	sM = (triDistP2[0]*triAlongSideP2P0Unit[0]) + (triDistP2[1]*triAlongSideP2P0Unit[1]) + (triDistP2[2]*triAlongSideP2P0Unit[2]);
	sP = (triDistP0[0]*triAlongSideP2P0Unit[0]) + (triDistP0[1]*triAlongSideP2P0Unit[1]) + (triDistP0[2]*triAlongSideP2P0Unit[2]);

	if ( (countCross==1) && (lineIndex==2) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			iL += ( t2 * (log( logArgNom )-log( LogArgTaylor(sM, dist) )) );
		}
	}

	if( lineIndex != 2 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		iL += ( t2 * (log(logArgNom)-log(logArgDenom)) );
	}

	return iL;
}

KThreeVector KElectrostaticRWGTriangleIntegrator::IqLField( const double* data, const double* P,
		const unsigned short countCross, const unsigned short lineIndex, const double dist ) const
{
	// function computes first term of equation (74) in the case q = -1

    // corner points P0, P1, P2 and P3

    const double triP0[3] = {
    		data[2], data[3], data[4] };

    const double triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const double triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on triangle surface

    double triN3[3];
    triN3[0] = data[6]*data[10] - data[7]*data[9];
    triN3[1] = data[7]*data[8]  - data[5]*data[10];
    triN3[2] = data[5]*data[9]  - data[6]*data[8];
    const double triMagN3 = 1./sqrt( POW2(triN3[0])+POW2(triN3[1])+POW2(triN3[2]) );
    triN3[0] = triN3[0]*triMagN3;
    triN3[1] = triN3[1]*triMagN3;
    triN3[2] = triN3[2]*triMagN3;

    // side line unit vectors

    double triAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    double triAlongSideP1P2Unit[3];

	const double magP1P2 = 1./sqrt( POW2(triP2[0]-triP1[0])
			+ POW2(triP2[1]-triP1[1])
			+ POW2(triP2[2]-triP1[2]) );
	triAlongSideP1P2Unit[0] = magP1P2 * (triP2[0]-triP1[0]);
	triAlongSideP1P2Unit[1] = magP1P2 * (triP2[1]-triP1[1]);
	triAlongSideP1P2Unit[2] = magP1P2 * (triP2[2]-triP1[2]);

    double triAlongSideP2P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

    // outward pointing vector m, perpendicular to side lines

    const double m0[3] = {
    		(triAlongSideP0P1Unit[1]*triN3[2]) - (triAlongSideP0P1Unit[2]*triN3[1]),
			(triAlongSideP0P1Unit[2]*triN3[0]) - (triAlongSideP0P1Unit[0]*triN3[2]),
			(triAlongSideP0P1Unit[0]*triN3[1]) - (triAlongSideP0P1Unit[1]*triN3[0])
    };

    const double m1[3] = {
    		(triAlongSideP1P2Unit[1]*triN3[2]) - (triAlongSideP1P2Unit[2]*triN3[1]),
			(triAlongSideP1P2Unit[2]*triN3[0]) - (triAlongSideP1P2Unit[0]*triN3[2]),
			(triAlongSideP1P2Unit[0]*triN3[1]) - (triAlongSideP1P2Unit[1]*triN3[0])
    };

    const double m2[3] = {
    		(triAlongSideP2P0Unit[1]*triN3[2]) - (triAlongSideP2P0Unit[2]*triN3[1]),
			(triAlongSideP2P0Unit[2]*triN3[0]) - (triAlongSideP2P0Unit[0]*triN3[2]),
			(triAlongSideP2P0Unit[0]*triN3[1]) - (triAlongSideP2P0Unit[1]*triN3[0])
    };

    // distance between triangle vertex points and field points in positive rotation order
    // pointing to the triangle vertex point

    const double triDistP0[3] = {
    		triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };

    const double triDistP1[3] = {
    		triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };

    const double triDistP2[3] = {
    		triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };

    const double triMagDistP0 = sqrt( POW2(triDistP0[0]) + POW2(triDistP0[1]) + POW2(triDistP0[2]) );
    const double triMagDistP1 = sqrt( POW2(triDistP1[0]) + POW2(triDistP1[1]) + POW2(triDistP1[2]) );
    const double triMagDistP2 = sqrt( POW2(triDistP2[0]) + POW2(triDistP2[1]) + POW2(triDistP2[2]) );

    // evaluation of line integral

    double logArgNom, logArgDenom;
	double iL[3] = {0., 0., 0.};
    double tmpScalar;

	// 0 //

    double rM = triMagDistP0;
    double rP = triMagDistP1;
    double sM = (triDistP0[0]*triAlongSideP0P1Unit[0]) + (triDistP0[1]*triAlongSideP0P1Unit[1]) + (triDistP0[2]*triAlongSideP0P1Unit[2]);
    double sP = (triDistP1[0]*triAlongSideP0P1Unit[0]) + (triDistP1[1]*triAlongSideP0P1Unit[1]) + (triDistP1[2]*triAlongSideP0P1Unit[2]);

	if ( (countCross==1) && (lineIndex==0) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = log( logArgNom ) - log( LogArgTaylor(sM, dist) );
			iL[0] += ( m0[0] * tmpScalar );
			iL[1] += ( m0[1] * tmpScalar );
			iL[2] += ( m0[2] * tmpScalar );
		}
	}

	if( lineIndex != 0 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		tmpScalar = (log(logArgNom)-log(logArgDenom));
		iL[0] += ( m0[0] * tmpScalar );
		iL[1] += ( m0[1] * tmpScalar );
		iL[2] += ( m0[2] * tmpScalar );
	}

	// 1 //

	rM = triMagDistP1;
	rP = triMagDistP2;
	sM = (triDistP1[0]*triAlongSideP1P2Unit[0]) + (triDistP1[1]*triAlongSideP1P2Unit[1]) + (triDistP1[2]*triAlongSideP1P2Unit[2]);
	sP = (triDistP2[0]*triAlongSideP1P2Unit[0]) + (triDistP2[1]*triAlongSideP1P2Unit[1]) + (triDistP2[2]*triAlongSideP1P2Unit[2]);

	if ( (countCross==1) && (lineIndex==1) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = log( logArgNom ) - log( LogArgTaylor(sM, dist) );
			iL[0] += ( m1[0] * tmpScalar );
			iL[1] += ( m1[1] * tmpScalar );
			iL[2] += ( m1[2] * tmpScalar );
		}
	}

	if( lineIndex != 1 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		tmpScalar = (log(logArgNom)-log(logArgDenom));
		iL[0] += ( m1[0] * tmpScalar );
		iL[1] += ( m1[1] * tmpScalar );
		iL[2] += ( m1[2] * tmpScalar );
	}

	// 2 //

	rM = triMagDistP2;
	rP = triMagDistP0;
	sM = (triDistP2[0]*triAlongSideP2P0Unit[0]) + (triDistP2[1]*triAlongSideP2P0Unit[1]) + (triDistP2[2]*triAlongSideP2P0Unit[2]);
	sP = (triDistP0[0]*triAlongSideP2P0Unit[0]) + (triDistP0[1]*triAlongSideP2P0Unit[1]) + (triDistP0[2]*triAlongSideP2P0Unit[2]);

	if ( (countCross==1) && (lineIndex==2) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = log( logArgNom ) - log( LogArgTaylor(sM, dist) );
			iL[0] += ( m2[0] * tmpScalar );
			iL[1] += ( m2[1] * tmpScalar );
			iL[2] += ( m2[2] * tmpScalar );
		}
	}

	if( lineIndex != 2 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		tmpScalar = (log(logArgNom)-log(logArgDenom));
		iL[0] += ( m2[0] * tmpScalar );
		iL[1] += ( m2[1] * tmpScalar );
		iL[2] += ( m2[2] * tmpScalar );
	}

	return KThreeVector(iL[0], iL[1], iL[2]);
}

std::pair<KThreeVector, double> KElectrostaticRWGTriangleIntegrator::IqLFieldAndPotential( const double* data, const double* P,
		const unsigned short countCross, const unsigned short lineIndex, const double dist ) const
{
    // corner points P0, P1 and P2

    const double triP0[3] = { data[2], data[3], data[4] };

    const double triP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const double triP2[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on triangle surface

    double triN3[3];
    triN3[0] = data[6]*data[10] - data[7]*data[9];
    triN3[1] = data[7]*data[8]  - data[5]*data[10];
    triN3[2] = data[5]*data[9]  - data[6]*data[8];
    const double triMagN3 = 1./sqrt( POW2(triN3[0])+POW2(triN3[1])+POW2(triN3[2]) );
    triN3[0] = triN3[0]*triMagN3;
    triN3[1] = triN3[1]*triMagN3;
    triN3[2] = triN3[2]*triMagN3;

    // side line unit vectors

    double triAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    double triAlongSideP1P2Unit[3];

	const double magP1P2 = 1./sqrt( POW2(triP2[0]-triP1[0])
			+ POW2(triP2[1]-triP1[1])
			+ POW2(triP2[2]-triP1[2]) );
	triAlongSideP1P2Unit[0] = magP1P2 * (triP2[0]-triP1[0]);
	triAlongSideP1P2Unit[1] = magP1P2 * (triP2[1]-triP1[1]);
	triAlongSideP1P2Unit[2] = magP1P2 * (triP2[2]-triP1[2]);

    double triAlongSideP2P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

	// length values of side lines, only half value is needed

    const double triAlongSideHalfLengthP0P1 = 0.5 * data[0];
    const double triAlongSideHalfLengthP1P2 = 0.5 * sqrt( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );
	const double triAlongSideHalfLengthP2P0 = 0.5 * data[1];

    // center point of each side to field point

    const double e0[3] = {
    		triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[0] + triP0[0],
			triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[1] + triP0[1],
			triAlongSideHalfLengthP0P1*triAlongSideP0P1Unit[2] + triP0[2] };

    const double e1[3] = {
    		triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[0] + triP1[0],
			triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[1] + triP1[1],
			triAlongSideHalfLengthP1P2*triAlongSideP1P2Unit[2] + triP1[2] };

    const double e2[3] = {
    		triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[0] + triP2[0],
			triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[1] + triP2[1],
			triAlongSideHalfLengthP2P0*triAlongSideP2P0Unit[2] + triP2[2] };

    // outward pointing vector m, perpendicular to side lines

    const double m0[3] = {
    		(triAlongSideP0P1Unit[1]*triN3[2]) - (triAlongSideP0P1Unit[2]*triN3[1]),
			(triAlongSideP0P1Unit[2]*triN3[0]) - (triAlongSideP0P1Unit[0]*triN3[2]),
			(triAlongSideP0P1Unit[0]*triN3[1]) - (triAlongSideP0P1Unit[1]*triN3[0])
    };

    const double m1[3] = {
    		(triAlongSideP1P2Unit[1]*triN3[2]) - (triAlongSideP1P2Unit[2]*triN3[1]),
			(triAlongSideP1P2Unit[2]*triN3[0]) - (triAlongSideP1P2Unit[0]*triN3[2]),
			(triAlongSideP1P2Unit[0]*triN3[1]) - (triAlongSideP1P2Unit[1]*triN3[0])
    };

    const double m2[3] = {
    		(triAlongSideP2P0Unit[1]*triN3[2]) - (triAlongSideP2P0Unit[2]*triN3[1]),
			(triAlongSideP2P0Unit[2]*triN3[0]) - (triAlongSideP2P0Unit[0]*triN3[2]),
			(triAlongSideP2P0Unit[0]*triN3[1]) - (triAlongSideP2P0Unit[1]*triN3[0])
    };

    // size t

    const double t0 = (m0[0]*(P[0]-e0[0])) + (m0[1]*(P[1]-e0[1])) + (m0[2]*(P[2]-e0[2]));
    const double t1 = (m1[0]*(P[0]-e1[0])) + (m1[1]*(P[1]-e1[1])) + (m1[2]*(P[2]-e1[2]));
    const double t2 = (m2[0]*(P[0]-e2[0])) + (m2[1]*(P[1]-e2[1])) + (m2[2]*(P[2]-e2[2]));

    // distance between triangle vertex points and field points in positive rotation order
    // pointing to the triangle vertex point

    const double triDistP0[3] = {
    		triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };

    const double triDistP1[3] = {
    		triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };

    const double triDistP2[3] = {
    		triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };

    const double triMagDistP0 = sqrt( POW2(triDistP0[0]) + POW2(triDistP0[1]) + POW2(triDistP0[2]) );
    const double triMagDistP1 = sqrt( POW2(triDistP1[0]) + POW2(triDistP1[1]) + POW2(triDistP1[2]) );
    const double triMagDistP2 = sqrt( POW2(triDistP2[0]) + POW2(triDistP2[1]) + POW2(triDistP2[2]) );

    // evaluation of line integral

    double logArgNom, logArgDenom;
    double iLPhi = 0.;
	double iLField[3] = {0., 0., 0.};
    double tmpScalar;

	// 0 //

    double rM = triMagDistP0;
    double rP = triMagDistP1;
    double sM = (triDistP0[0]*triAlongSideP0P1Unit[0]) + (triDistP0[1]*triAlongSideP0P1Unit[1]) + (triDistP0[2]*triAlongSideP0P1Unit[2]);
    double sP = (triDistP1[0]*triAlongSideP0P1Unit[0]) + (triDistP1[1]*triAlongSideP0P1Unit[1]) + (triDistP1[2]*triAlongSideP0P1Unit[2]);

	if ( (countCross==1) && (lineIndex==0) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = (log( logArgNom )-log( LogArgTaylor(sM, dist) ));
			iLField[0] += ( m0[0] * tmpScalar );
			iLField[1] += ( m0[1] * tmpScalar );
			iLField[2] += ( m0[2] * tmpScalar );
			iLPhi += ( t0 * tmpScalar );
		}
	}

	if( lineIndex != 0 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		tmpScalar = (log( logArgNom )-log( logArgDenom ));
		iLField[0] += ( m0[0] * tmpScalar );
		iLField[1] += ( m0[1] * tmpScalar );
		iLField[2] += ( m0[2] * tmpScalar );
		iLPhi += ( t0 * tmpScalar );
	}

	// 1 //

	rM = triMagDistP1;
	rP = triMagDistP2;
	sM = (triDistP1[0]*triAlongSideP1P2Unit[0]) + (triDistP1[1]*triAlongSideP1P2Unit[1]) + (triDistP1[2]*triAlongSideP1P2Unit[2]);
	sP = (triDistP2[0]*triAlongSideP1P2Unit[0]) + (triDistP2[1]*triAlongSideP1P2Unit[1]) + (triDistP2[2]*triAlongSideP1P2Unit[2]);

	if ( (countCross==1) && (lineIndex==1) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = (log( logArgNom )-log( LogArgTaylor(sM, dist) ));
			iLField[0] += ( m1[0] * tmpScalar );
			iLField[1] += ( m1[1] * tmpScalar );
			iLField[2] += ( m1[2] * tmpScalar );
			iLPhi += ( t1 * tmpScalar );
		}
	}

	if( lineIndex != 1 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		tmpScalar = (log( logArgNom )-log( logArgDenom ));
		iLField[0] += ( m1[0] * tmpScalar );
		iLField[1] += ( m1[1] * tmpScalar );
		iLField[2] += ( m1[2] * tmpScalar );
		iLPhi += ( t1 * tmpScalar );
	}

	// 2 //

	rM = triMagDistP2;
	rP = triMagDistP0;
	sM = (triDistP2[0]*triAlongSideP2P0Unit[0]) + (triDistP2[1]*triAlongSideP2P0Unit[1]) + (triDistP2[2]*triAlongSideP2P0Unit[2]);
	sP = (triDistP0[0]*triAlongSideP2P0Unit[0]) + (triDistP0[1]*triAlongSideP2P0Unit[1]) + (triDistP0[2]*triAlongSideP2P0Unit[2]);

	if ( (countCross==1) && (lineIndex==2) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = (log( logArgNom )-log( LogArgTaylor(sM, dist) ));
			iLField[0] += ( m2[0] * tmpScalar );
			iLField[1] += ( m2[1] * tmpScalar );
			iLField[2] += ( m2[2] * tmpScalar );
			iLPhi += ( t2 * tmpScalar );
		}
	}

	if( lineIndex != 2 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}

		tmpScalar = (log( logArgNom )-log( logArgDenom ));
		iLField[0] += ( m2[0] * tmpScalar );
		iLField[1] += ( m2[1] * tmpScalar );
		iLField[2] += ( m2[2] * tmpScalar );
		iLPhi += ( t2 * tmpScalar );
	}

	return std::make_pair( KThreeVector(iLField[0], iLField[1], iLField[2]), iLPhi );
}

double KElectrostaticRWGTriangleIntegrator::Potential( const KTriangle* source, const KPosition& P ) const
{
	// save triangle data into double array

	const double triData[11] = {
			source->GetA(),
			source->GetB(),
			source->GetP0().X(),
			source->GetP0().Y(),
			source->GetP0().Z(),
			source->GetN1().X(),
			source->GetN1().Y(),
			source->GetN1().Z(),
			source->GetN2().X(),
			source->GetN2().Y(),
			source->GetN2().Z()
	};

    // corner points P0, P1 and P2

    const double triP0[3] = {
    		triData[2], triData[3], triData[4] };

    const double triP1[3] = {
    		triData[2] + (triData[0]*triData[5]),
			triData[3] + (triData[0]*triData[6]),
			triData[4] + (triData[0]*triData[7]) }; // = fP0 + fN1*fA

    const double triP2[3] = {
    		triData[2] + (triData[1]*triData[8]),
			triData[3] + (triData[1]*triData[9]),
			triData[4] + (triData[1]*triData[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on triangle surface

    double triN3[3];
    triN3[0] = triData[6]*triData[10] - triData[7]*triData[9];
    triN3[1] = triData[7]*triData[8]  - triData[5]*triData[10];
    triN3[2] = triData[5]*triData[9]  - triData[6]*triData[8];
    const double triMagN3 = 1./sqrt(POW2(triN3[0]) + POW2(triN3[1]) + POW2(triN3[2]));
    triN3[0] = triN3[0]*triMagN3;
    triN3[1] = triN3[1]*triMagN3;
    triN3[2] = triN3[2]*triMagN3;

    // triangle centroid

    const double triCenter[3] = {
    		triData[2] + (triData[0]*triData[5] + triData[1]*triData[8])/3.,
			triData[3] + (triData[0]*triData[6] + triData[1]*triData[9])/3.,
			triData[4] + (triData[0]*triData[7] + triData[1]*triData[10])/3.};

    // side line vectors

    const double triAlongSideP0P1[3] = {
    		triData[0]*triData[5],
			triData[0]*triData[6],
			triData[0]*triData[7] }; // = A * N1

    const double triAlongSideP1P2[3] = {
    		triP2[0]-triP1[0],
			triP2[1]-triP1[1],
			triP2[2]-triP1[2] };

    const double triAlongSideP2P0[3] = {
    		(-1)*triData[1]*triData[8],
			(-1)*triData[1]*triData[9],
			(-1)*triData[1]*triData[10] }; // = -B * N2

    // length values of side lines

    const double triAlongSideLengthP0P1 = triData[0];
    const double triAlongSideLengthP1P2 = sqrt( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );
	const double triAlongSideLengthP2P0 = triData[1];

	// distance between triangle vertex points and field points in positive rotation order
	// pointing to the triangle vertex point

	const double triDistP0[3] = {
			triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };
	const double triDistP1[3] = {
			triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };
	const double triDistP2[3] = {
			triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };

	// check if field point is at triangle edge or at side line

    double distToLine = 0.;
    double distToLineMin = 0.;

    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on triangle edge (= 2) or point is on side line (= 1)
    double lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    double tmpVector[3];
    double tmpScalar;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./triAlongSideLengthP0P1;

    // compute cross product

    tmpVector[0] = (triAlongSideP0P1[1]*triDistP0[2]) - (triAlongSideP0P1[2]*triDistP0[1]);
    tmpVector[1] = (triAlongSideP0P1[2]*triDistP0[0]) - (triAlongSideP0P1[0]*triDistP0[2]);
    tmpVector[2] = (triAlongSideP0P1[0]*triDistP0[1]) - (triAlongSideP0P1[1]*triDistP0[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0
    lineLambda = ((-triDistP0[0]*triAlongSideP0P1[0]) + (-triDistP0[1]*triAlongSideP0P1[1]) + (-triDistP0[2]*triAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./triAlongSideLengthP1P2;

    // compute cross product

    tmpVector[0] = (triAlongSideP1P2[1]*triDistP1[2]) - (triAlongSideP1P2[2]*triDistP1[1]);
    tmpVector[1] = (triAlongSideP1P2[2]*triDistP1[0]) - (triAlongSideP1P2[0]*triDistP1[2]);
    tmpVector[2] = (triAlongSideP1P2[0]*triDistP1[1]) - (triAlongSideP1P2[1]*triDistP1[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction triDistP1 vector

    lineLambda = ((-triDistP1[0]*triAlongSideP1P2[0]) + (-triDistP1[1]*triAlongSideP1P2[1]) + (-triDistP1[2]*triAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P0 side line

    tmpScalar = 1./triAlongSideLengthP2P0;

    // compute cross product

    tmpVector[0] = (triAlongSideP2P0[1]*triDistP2[2]) - (triAlongSideP2P0[2]*triDistP2[1]);
    tmpVector[1] = (triAlongSideP2P0[2]*triDistP2[0]) - (triAlongSideP2P0[0]*triDistP2[2]);
    tmpVector[2] = (triAlongSideP2P0[0]*triDistP2[1]) - (triAlongSideP2P0[1]*triDistP2[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for triDistP2

    lineLambda = ((-triDistP2[0]*triAlongSideP2P0[0]) + (-triDistP2[1]*triAlongSideP2P0[1]) + (-triDistP2[2]*triAlongSideP2P0[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length 'CORRECTIONTRIN3' in positive and negative N3 direction

    if( correctionCounter == 2 ) {
    	const double upEps[3] = {
    			P[0] + fDistanceCorrectionN3*triN3[0],
				P[1] + fDistanceCorrectionN3*triN3[1],
				P[2] + fDistanceCorrectionN3*triN3[2]
    	};

    	const double downEps[3] = {
    			P[0] - fDistanceCorrectionN3*triN3[0],
				P[1] - fDistanceCorrectionN3*triN3[1],
				P[2] - fDistanceCorrectionN3*triN3[2]
    	};

    	// compute IqS term

        const double hUp = ( triN3[0] * (upEps[0]-triCenter[0]) )
				+ ( triN3[1] * (upEps[1]-triCenter[1]) )
				+ ( triN3[2] * (upEps[2]-triCenter[2]) );

        const double solidAngleUp = solidAngle.SolidAngleTriangleAsArray( triData, upEps );

        const double hDown = ( triN3[0] * (downEps[0]-triCenter[0]) )
				+ ( triN3[1] * (downEps[1]-triCenter[1]) )
				+ ( triN3[2] * (downEps[2]-triCenter[2]) );

        const double solidAngleDown = solidAngle.SolidAngleTriangleAsArray( triData, downEps );

    	// compute IqL

    	const double IqLUp = IqLPotential( triData, upEps, 9, 9, 9 ); /* no line correction */

    	const double IqLDown = IqLPotential( triData, downEps, 9, 9, 9 ); /* no line correction */

    	const double finalResult = 0.5*((-hUp*solidAngleUp - IqLUp) + (-hDown*solidAngleDown - IqLDown));

    	return finalResult*KEMConstants::OneOverFourPiEps0;
    }

    const double h = ( triN3[0] * (P[0]-triCenter[0]) )
			+ ( triN3[1] * (P[1]-triCenter[1]) )
			+ ( triN3[2] * (P[2]-triCenter[2]) );

    double fieldPoint[3] = { P[0], P[1], P[2] };

    const double triSolidAngle = solidAngle.SolidAngleTriangleAsArray( triData, fieldPoint );

	const double finalResult = (-h*triSolidAngle) - IqLPotential( triData, fieldPoint, correctionCounter, correctionLineIndex, distToLineMin );

    return finalResult*KEMConstants::OneOverFourPiEps0;
}



KThreeVector KElectrostaticRWGTriangleIntegrator::ElectricField( const KTriangle* source, const KPosition& P ) const
{
	// save triangle data into double array

	const double triData[11] = {
			source->GetA(),
			source->GetB(),
			source->GetP0().X(),
			source->GetP0().Y(),
			source->GetP0().Z(),
			source->GetN1().X(),
			source->GetN1().Y(),
			source->GetN1().Z(),
			source->GetN2().X(),
			source->GetN2().Y(),
			source->GetN2().Z()
	};

	// get perpendicular normal vector n3 on triangle surface

    double triN3[3];
    triN3[0] = triData[6]*triData[10] - triData[7]*triData[9];
    triN3[1] = triData[7]*triData[8]  - triData[5]*triData[10];
    triN3[2] = triData[5]*triData[9]  - triData[6]*triData[8];
    const double triMagN3 = 1./sqrt(POW2(triN3[0]) + POW2(triN3[1]) + POW2(triN3[2]));
    triN3[0] = triN3[0]*triMagN3;
    triN3[1] = triN3[1]*triMagN3;
    triN3[2] = triN3[2]*triMagN3;

    // corner points P0, P1 and P2

    const double triP0[3] = {
    		triData[2], triData[3], triData[4] };

    const double triP1[3] = {
    		triData[2] + (triData[0]*triData[5]),
			triData[3] + (triData[0]*triData[6]),
			triData[4] + (triData[0]*triData[7]) }; // = fP0 + fN1*fA

    const double triP2[3] = {
    		triData[2] + (triData[1]*triData[8]),
			triData[3] + (triData[1]*triData[9]),
			triData[4] + (triData[1]*triData[10]) }; // = fP0 + fN2*fB

    // side line vectors

    const double triAlongSideP0P1[3] = {
    		triData[0]*triData[5],
			triData[0]*triData[6],
			triData[0]*triData[7] }; // = A * N1

    const double triAlongSideP1P2[3] = {
    		triP2[0]-triP1[0],
			triP2[1]-triP1[1],
			triP2[2]-triP1[2] };

    const double triAlongSideP2P0[3] = {
    		(-1)*triData[1]*triData[8],
			(-1)*triData[1]*triData[9],
			(-1)*triData[1]*triData[10] }; // = -B * N2

    // length values of side lines

    const double triAlongSideLengthP0P1 = triData[0];
    const double triAlongSideLengthP1P2 = sqrt( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );
	const double triAlongSideLengthP2P0 = triData[1];

	// distance between triangle vertex points and field points in positive rotation order
	// pointing to the triangle vertex point

	const double triDistP0[3] = {
			triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };
	const double triDistP1[3] = {
			triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };
	const double triDistP2[3] = {
			triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };

	// check if field point is at triangle edge or at side line

    double distToLine = 0.;
    double distToLineMin = 0.;

    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on triangle edge (= 2) or point is on side line (= 1)
    double lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    double tmpVector[3];
    double tmpScalar;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./triAlongSideLengthP0P1;

    // compute cross product

    tmpVector[0] = (triAlongSideP0P1[1]*triDistP0[2]) - (triAlongSideP0P1[2]*triDistP0[1]);
    tmpVector[1] = (triAlongSideP0P1[2]*triDistP0[0]) - (triAlongSideP0P1[0]*triDistP0[2]);
    tmpVector[2] = (triAlongSideP0P1[0]*triDistP0[1]) - (triAlongSideP0P1[1]*triDistP0[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0

    lineLambda = ((-triDistP0[0]*triAlongSideP0P1[0]) + (-triDistP0[1]*triAlongSideP0P1[1]) + (-triDistP0[2]*triAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./triAlongSideLengthP1P2;

    // compute cross product

    tmpVector[0] = (triAlongSideP1P2[1]*triDistP1[2]) - (triAlongSideP1P2[2]*triDistP1[1]);
    tmpVector[1] = (triAlongSideP1P2[2]*triDistP1[0]) - (triAlongSideP1P2[0]*triDistP1[2]);
    tmpVector[2] = (triAlongSideP1P2[0]*triDistP1[1]) - (triAlongSideP1P2[1]*triDistP1[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction triDistP1 vector

    lineLambda = ((-triDistP1[0]*triAlongSideP1P2[0]) + (-triDistP1[1]*triAlongSideP1P2[1]) + (-triDistP1[2]*triAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P0 side line

    tmpScalar = 1./triAlongSideLengthP2P0;

    // compute cross product

    tmpVector[0] = (triAlongSideP2P0[1]*triDistP2[2]) - (triAlongSideP2P0[2]*triDistP2[1]);
    tmpVector[1] = (triAlongSideP2P0[2]*triDistP2[0]) - (triAlongSideP2P0[0]*triDistP2[2]);
    tmpVector[2] = (triAlongSideP2P0[0]*triDistP2[1]) - (triAlongSideP2P0[1]*triDistP2[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for triDistP2

    lineLambda = ((-triDistP2[0]*triAlongSideP2P0[0]) + (-triDistP2[1]*triAlongSideP2P0[1]) + (-triDistP2[2]*triAlongSideP2P0[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    if( correctionCounter == 2 ) {
    	const double upEps[3] = {
    			P[0] + fDistanceCorrectionN3*triN3[0],
				P[1] + fDistanceCorrectionN3*triN3[1],
				P[2] + fDistanceCorrectionN3*triN3[2]
    	};
    	const double downEps[3] = {
    			P[0] - fDistanceCorrectionN3*triN3[0],
				P[1] - fDistanceCorrectionN3*triN3[1],
				P[2] - fDistanceCorrectionN3*triN3[2]
    	};

    	// compute IqS term

        const double solidAngleUp
			= solidAngle.SolidAngleTriangleAsArray( triData, upEps );

        const double solidAngleDown
			= solidAngle.SolidAngleTriangleAsArray( triData, downEps );

    	// compute IqL

    	const KThreeVector IqLUp
			= IqLField( triData, upEps, 9, 9, 9 ); /* no line correction */
    	const KThreeVector IqLDown
			= IqLField( triData, downEps, 9, 9, 9 ); /* no line correction */

    	const KThreeVector finalResult(
    			(triN3[0]*solidAngleUp + IqLUp[0]) + (triN3[0]*solidAngleDown + IqLDown[0]),
				(triN3[1]*solidAngleUp + IqLUp[1]) + (triN3[1]*solidAngleDown + IqLDown[1]),
				(triN3[2]*solidAngleUp + IqLUp[2]) + (triN3[2]*solidAngleDown + IqLDown[2]) );

    	return (0.5*finalResult)*KEMConstants::OneOverFourPiEps0;
    }

    double fieldPoint[3] = { P[0], P[1], P[2] };

    const double triSolidAngle = solidAngle.SolidAngleTriangleAsArray( triData, fieldPoint );

    const KThreeVector IqLEField
		= IqLField( triData, fieldPoint, correctionCounter, correctionLineIndex, distToLineMin );

    const KThreeVector finalResult (
    		(triN3[0]*triSolidAngle + IqLEField.X()),
    		(triN3[1]*triSolidAngle + IqLEField.Y()),
			(triN3[2]*triSolidAngle + IqLEField.Z()) );

    return finalResult*KEMConstants::OneOverFourPiEps0;
}

std::pair<KThreeVector, double> KElectrostaticRWGTriangleIntegrator::ElectricFieldAndPotential( const KTriangle* source, const KPosition& P ) const
{
	// save triangle data into double array

	const double triData[11] = {
			source->GetA(),
			source->GetB(),
			source->GetP0().X(),
			source->GetP0().Y(),
			source->GetP0().Z(),
			source->GetN1().X(),
			source->GetN1().Y(),
			source->GetN1().Z(),
			source->GetN2().X(),
			source->GetN2().Y(),
			source->GetN2().Z()
	};

    // corner points P0, P1 and P2

    const double triP0[3] = {
    		triData[2], triData[3], triData[4] };

    const double triP1[3] = {
    		triData[2] + (triData[0]*triData[5]),
			triData[3] + (triData[0]*triData[6]),
			triData[4] + (triData[0]*triData[7]) }; // = fP0 + fN1*fA

    const double triP2[3] = {
    		triData[2] + (triData[1]*triData[8]),
			triData[3] + (triData[1]*triData[9]),
			triData[4] + (triData[1]*triData[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on triangle surface

    double triN3[3];
    triN3[0] = triData[6]*triData[10] - triData[7]*triData[9];
    triN3[1] = triData[7]*triData[8]  - triData[5]*triData[10];
    triN3[2] = triData[5]*triData[9]  - triData[6]*triData[8];
    const double triMagN3 = 1./sqrt(POW2(triN3[0]) + POW2(triN3[1]) + POW2(triN3[2]));
    triN3[0] = triN3[0]*triMagN3;
    triN3[1] = triN3[1]*triMagN3;
    triN3[2] = triN3[2]*triMagN3;

    // triangle centroid

    const double triCenter[3] = {
    		triData[2] + (triData[0]*triData[5] + triData[1]*triData[8])/3.,
			triData[3] + (triData[0]*triData[6] + triData[1]*triData[9])/3.,
			triData[4] + (triData[0]*triData[7] + triData[1]*triData[10])/3.};

    // side line vectors

    const double triAlongSideP0P1[3] = {
    		triData[0]*triData[5],
			triData[0]*triData[6],
			triData[0]*triData[7] }; // = A * N1

    const double triAlongSideP1P2[3] = {
    		triP2[0]-triP1[0],
			triP2[1]-triP1[1],
			triP2[2]-triP1[2] };

    const double triAlongSideP2P0[3] = {
    		(-1)*triData[1]*triData[8],
			(-1)*triData[1]*triData[9],
			(-1)*triData[1]*triData[10] }; // = -B * N2

    // length values of side lines

    const double triAlongSideLengthP0P1 = triData[0];
    const double triAlongSideLengthP1P2 = sqrt( POW2(triP2[0]-triP1[0]) + POW2(triP2[1]-triP1[1]) + POW2(triP2[2]-triP1[2]) );
	const double triAlongSideLengthP2P0 = triData[1];

	// distance between triangle vertex points and field points in positive rotation order
	// pointing to the triangle vertex point

	const double triDistP0[3] = {
			triP0[0] - P[0],
			triP0[1] - P[1],
			triP0[2] - P[2] };
	const double triDistP1[3] = {
			triP1[0] - P[0],
			triP1[1] - P[1],
			triP1[2] - P[2] };
	const double triDistP2[3] = {
			triP2[0] - P[0],
			triP2[1] - P[1],
			triP2[2] - P[2] };

	// check if field point is at triangle edge or at side line

    double distToLine = 0.;
    double distToLineMin = 0.;

    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on triangle edge (= 2) or point is on side line (= 1)
    double lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    double tmpVector[3];
    double tmpScalar;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./triAlongSideLengthP0P1;

    // compute cross product
    tmpVector[0] = (triAlongSideP0P1[1]*triDistP0[2]) - (triAlongSideP0P1[2]*triDistP0[1]);
    tmpVector[1] = (triAlongSideP0P1[2]*triDistP0[0]) - (triAlongSideP0P1[0]*triDistP0[2]);
    tmpVector[2] = (triAlongSideP0P1[0]*triDistP0[1]) - (triAlongSideP0P1[1]*triDistP0[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0
    lineLambda = ((-triDistP0[0]*triAlongSideP0P1[0]) + (-triDistP0[1]*triAlongSideP0P1[1]) + (-triDistP0[2]*triAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./triAlongSideLengthP1P2;

    // compute cross product

    tmpVector[0] = (triAlongSideP1P2[1]*triDistP1[2]) - (triAlongSideP1P2[2]*triDistP1[1]);
    tmpVector[1] = (triAlongSideP1P2[2]*triDistP1[0]) - (triAlongSideP1P2[0]*triDistP1[2]);
    tmpVector[2] = (triAlongSideP1P2[0]*triDistP1[1]) - (triAlongSideP1P2[1]*triDistP1[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction triDistP1 vector

    lineLambda = ((-triDistP1[0]*triAlongSideP1P2[0]) + (-triDistP1[1]*triAlongSideP1P2[1]) + (-triDistP1[2]*triAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P0 side line

    tmpScalar = 1./triAlongSideLengthP2P0;

    // compute cross product

    tmpVector[0] = (triAlongSideP2P0[1]*triDistP2[2]) - (triAlongSideP2P0[2]*triDistP2[1]);
    tmpVector[1] = (triAlongSideP2P0[2]*triDistP2[0]) - (triAlongSideP2P0[0]*triDistP2[2]);
    tmpVector[2] = (triAlongSideP2P0[0]*triDistP2[1]) - (triAlongSideP2P0[1]*triDistP2[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for triDistP2

    lineLambda = ((-triDistP2[0]*triAlongSideP2P0[0]) + (-triDistP2[1]*triAlongSideP2P0[1]) + (-triDistP2[2]*triAlongSideP2P0[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length 'CORRECTIONTRIN3' in positive and negative N3 direction

    if( correctionCounter == 2 ) {
    	const double upEps[3] = {
    			P[0] + fDistanceCorrectionN3*triN3[0],
				P[1] + fDistanceCorrectionN3*triN3[1],
				P[2] + fDistanceCorrectionN3*triN3[2]
    	};
    	const double downEps[3] = {
    			P[0] - fDistanceCorrectionN3*triN3[0],
				P[1] - fDistanceCorrectionN3*triN3[1],
				P[2] - fDistanceCorrectionN3*triN3[2]
    	};

    	// compute IqS term

        const double hUp = ( triN3[0] * (upEps[0]-triCenter[0]) )
				+ ( triN3[1] * (upEps[1]-triCenter[1]) )
				+ ( triN3[2] * (upEps[2]-triCenter[2]) );

        const double solidAngleUp = solidAngle.SolidAngleTriangleAsArray( triData, upEps );

        const double hDown = ( triN3[0] * (downEps[0]-triCenter[0]) )
				+ ( triN3[1] * (downEps[1]-triCenter[1]) )
				+ ( triN3[2] * (downEps[2]-triCenter[2]) );

        const double solidAngleDown = solidAngle.SolidAngleTriangleAsArray( triData, downEps );

    	// compute IqL

        std::pair<KThreeVector, double> IqLFieldAndPotentialUp
			= IqLFieldAndPotential( triData, upEps, 9, 9, 9 ); /* no line correction */

        std::pair<KThreeVector, double> IqLFieldAndPotentialDown
			= IqLFieldAndPotential( triData, downEps, 9, 9, 9 ); /* no line correction */

    	const KThreeVector finalField(
    			KEMConstants::OneOverFourPiEps0 * 0.5 * ((triN3[0]*solidAngleUp + IqLFieldAndPotentialUp.first[0]) + (triN3[0]*solidAngleDown + IqLFieldAndPotentialDown.first[0])),
				KEMConstants::OneOverFourPiEps0 * 0.5 * ((triN3[1]*solidAngleUp + IqLFieldAndPotentialUp.first[1]) + (triN3[1]*solidAngleDown + IqLFieldAndPotentialDown.first[1])),
				KEMConstants::OneOverFourPiEps0 * 0.5 * ((triN3[2]*solidAngleUp + IqLFieldAndPotentialUp.first[2]) + (triN3[2]*solidAngleDown + IqLFieldAndPotentialDown.first[2])) );

    	double finalPotential = KEMConstants::OneOverFourPiEps0*0.5 * (
    			(-hUp*solidAngleUp - IqLFieldAndPotentialUp.second)
				+ (-hDown*solidAngleDown - IqLFieldAndPotentialDown.second) );

    	return std::make_pair( finalField, finalPotential );

    }

    const double h = ( triN3[0] * (P[0]-triCenter[0]) )
			+ ( triN3[1] * (P[1]-triCenter[1]) )
			+ ( triN3[2] * (P[2]-triCenter[2]) );

    double fieldPoint[3] = { P[0], P[1], P[2] };

    const double triSolidAngle = solidAngle.SolidAngleTriangleAsArray( triData, fieldPoint );

    std::pair<KThreeVector, double> IqLFieldAndPhi
		= IqLFieldAndPotential( triData, fieldPoint, correctionCounter, correctionLineIndex, distToLineMin );

	const KThreeVector finalField(
			KEMConstants::OneOverFourPiEps0*(triN3[0]*triSolidAngle + IqLFieldAndPhi.first[0]),
			KEMConstants::OneOverFourPiEps0*(triN3[1]*triSolidAngle + IqLFieldAndPhi.first[1]),
			KEMConstants::OneOverFourPiEps0*(triN3[2]*triSolidAngle + IqLFieldAndPhi.first[2]) );
	const double finalPhi = KEMConstants::OneOverFourPiEps0*((-h*triSolidAngle) - IqLFieldAndPhi.second);

    return std::make_pair( finalField, finalPhi );
}


double KElectrostaticRWGTriangleIntegrator::Potential(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const
{
    double potential( 0. );

    for( KSymmetryGroup<KTriangle>::ShapeCIt it=source->begin(); it!=source->end(); ++it )
        potential += Potential(*it,P);

    return potential;
}


KThreeVector KElectrostaticRWGTriangleIntegrator::ElectricField(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const
{
    KThreeVector electricField( 0., 0., 0. );

    for( KSymmetryGroup<KTriangle>::ShapeCIt it=source->begin(); it!=source->end(); ++it )
        electricField += ElectricField(*it,P);

    return electricField;
}

std::pair<KThreeVector, double> KElectrostaticRWGTriangleIntegrator::ElectricFieldAndPotential(const KSymmetryGroup<KTriangle>* source, const KPosition& P) const
{
	std::pair<KThreeVector, double> fieldAndPotential;
    double potential( 0. );
    KThreeVector electricField( 0., 0., 0. );

    for( KSymmetryGroup<KTriangle>::ShapeCIt it=source->begin(); it!=source->end(); ++it ) {
    	fieldAndPotential = ElectricFieldAndPotential( *it, P );
        electricField += fieldAndPotential.first;
    	potential += fieldAndPotential.second;
    }

    return std::make_pair( electricField, potential );
}


}
