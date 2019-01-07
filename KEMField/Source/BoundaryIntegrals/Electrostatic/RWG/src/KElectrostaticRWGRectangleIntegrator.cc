#include "KElectrostaticRWGRectangleIntegrator.hh"

#define POW2(x) ((x)*(x))

namespace KEMField
{

double KElectrostaticRWGRectangleIntegrator::LogArgTaylor( const double sMin, const double dist ) const
{
	double quotient = fabs(dist/sMin);
	if( quotient < 1.e-14 ) quotient = 1.e-14;

	// Taylor expansion of log argument to second order
	double res = 0.5*fabs(sMin)*POW2(quotient);

	return res;
}

double KElectrostaticRWGRectangleIntegrator::IqLPotential( const double* data, const double* P,
		const unsigned short countCross, const unsigned short lineIndex, const double dist ) const
{
	// computes second term of equation (63) in the case q = -1

    // corner points P0, P1, P2 and P3

    const double rectP0[3] = {
    		data[2], data[3], data[4] };

    const double rectP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const double rectP2[3] = {
    		data[2] + (data[0]*data[5]) + (data[1]*data[8]),
    		data[3] + (data[0]*data[6]) + (data[1]*data[9]),
			data[4] + (data[0]*data[7]) + (data[1]*data[10]) }; // = fP0 + fN1*fA + fN2*fB

    const double rectP3[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on rectangle surface

    double rectN3[3];
    rectN3[0] = data[6]*data[10] - data[7]*data[9];
    rectN3[1] = data[7]*data[8]  - data[5]*data[10];
    rectN3[2] = data[5]*data[9]  - data[6]*data[8];
    const double rectMagN3 = 1./sqrt( POW2(rectN3[0])+POW2(rectN3[1])+POW2(rectN3[2]) );
    rectN3[0] = rectN3[0]*rectMagN3;
    rectN3[1] = rectN3[1]*rectMagN3;
    rectN3[2] = rectN3[2]*rectMagN3;

    // side line unit vectors

    const double rectAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    const double rectAlongSideP1P2Unit[3] = {data[8], data[9], data[10]}; // = N2
    const double rectAlongSideP2P3Unit[3] = {-data[5], -data[6], -data[7]}; // = -N1
    const double rectAlongSideP3P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

    // center point of each side to field point

    const double e0[3] = {
    		(0.5*data[0]*data[5]) + rectP0[0],
			(0.5*data[0]*data[6]) + rectP0[1],
			(0.5*data[0]*data[7]) + rectP0[2] };

    const double e1[3] = {
    		(0.5*data[1]*data[8]) + rectP1[0],
			(0.5*data[1]*data[9]) + rectP1[1],
			(0.5*data[1]*data[10]) + rectP1[2] };

    const double e2[3] = {
    		(-0.5*data[0]*data[5]) + rectP2[0],
			(-0.5*data[0]*data[6]) + rectP2[1],
			(-0.5*data[0]*data[7]) + rectP2[2] };

    const double e3[3] = {
    		(-0.5*data[1]*data[8]) + rectP3[0],
			(-0.5*data[1]*data[9]) + rectP3[1],
			(-0.5*data[1]*data[10]) + rectP3[2] };

    // outward pointing vector m, perpendicular to side lines

    const double m0[3] = {
    		(rectAlongSideP0P1Unit[1]*rectN3[2]) - (rectAlongSideP0P1Unit[2]*rectN3[1]),
			(rectAlongSideP0P1Unit[2]*rectN3[0]) - (rectAlongSideP0P1Unit[0]*rectN3[2]),
			(rectAlongSideP0P1Unit[0]*rectN3[1]) - (rectAlongSideP0P1Unit[1]*rectN3[0])
    };

    const double m1[3] = {
    		(rectAlongSideP1P2Unit[1]*rectN3[2]) - (rectAlongSideP1P2Unit[2]*rectN3[1]),
			(rectAlongSideP1P2Unit[2]*rectN3[0]) - (rectAlongSideP1P2Unit[0]*rectN3[2]),
			(rectAlongSideP1P2Unit[0]*rectN3[1]) - (rectAlongSideP1P2Unit[1]*rectN3[0])
    };

    const double m2[3] = {
    		(rectAlongSideP2P3Unit[1]*rectN3[2]) - (rectAlongSideP2P3Unit[2]*rectN3[1]),
			(rectAlongSideP2P3Unit[2]*rectN3[0]) - (rectAlongSideP2P3Unit[0]*rectN3[2]),
			(rectAlongSideP2P3Unit[0]*rectN3[1]) - (rectAlongSideP2P3Unit[1]*rectN3[0])
    };

    const double m3[3] = {
    		(rectAlongSideP3P0Unit[1]*rectN3[2]) - (rectAlongSideP3P0Unit[2]*rectN3[1]),
			(rectAlongSideP3P0Unit[2]*rectN3[0]) - (rectAlongSideP3P0Unit[0]*rectN3[2]),
			(rectAlongSideP3P0Unit[0]*rectN3[1]) - (rectAlongSideP3P0Unit[1]*rectN3[0])
    };

    // size t

    const double t0 = (m0[0]*(P[0]-e0[0])) + (m0[1]*(P[1]-e0[1])) + (m0[2]*(P[2]-e0[2]));
    const double t1 = (m1[0]*(P[0]-e1[0])) + (m1[1]*(P[1]-e1[1])) + (m1[2]*(P[2]-e1[2]));
    const double t2 = (m2[0]*(P[0]-e2[0])) + (m2[1]*(P[1]-e2[1])) + (m2[2]*(P[2]-e2[2]));
    const double t3 = (m3[0]*(P[0]-e3[0])) + (m3[1]*(P[1]-e3[1])) + (m3[2]*(P[2]-e3[2]));

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const double rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const double rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const double rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const double rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };

   	const double rectDistP0Mag = sqrt( POW2(rectDistP0[0]) + POW2(rectDistP0[1]) + POW2(rectDistP0[2]) );
	const double rectDistP1Mag = sqrt( POW2(rectDistP1[0]) + POW2(rectDistP1[1]) + POW2(rectDistP1[2]) );
	const double rectDistP2Mag = sqrt( POW2(rectDistP2[0]) + POW2(rectDistP2[1]) + POW2(rectDistP2[2]) );
	const double rectDistP3Mag = sqrt( POW2(rectDistP3[0]) + POW2(rectDistP3[1]) + POW2(rectDistP3[2]) );

    // evaluation of line integral

    // 0 //

	double logArgNom, logArgDenom;
	double iL = 0.;

	double rM = rectDistP0Mag;
	double rP = rectDistP1Mag;
	double sM = (rectDistP0[0]*rectAlongSideP0P1Unit[0]) + (rectDistP0[1]*rectAlongSideP0P1Unit[1]) + (rectDistP0[2]*rectAlongSideP0P1Unit[2]);
	double sP = (rectDistP1[0]*rectAlongSideP0P1Unit[0]) + (rectDistP1[1]*rectAlongSideP0P1Unit[1]) + (rectDistP1[2]*rectAlongSideP0P1Unit[2]);

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

	rM = rectDistP1Mag;
	rP = rectDistP2Mag;
	sM = (rectDistP1[0]*rectAlongSideP1P2Unit[0]) + (rectDistP1[1]*rectAlongSideP1P2Unit[1]) + (rectDistP1[2]*rectAlongSideP1P2Unit[2]);
	sP = (rectDistP2[0]*rectAlongSideP1P2Unit[0]) + (rectDistP2[1]*rectAlongSideP1P2Unit[1]) + (rectDistP2[2]*rectAlongSideP1P2Unit[2]);

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

	rM = rectDistP2Mag;
	rP = rectDistP3Mag;
	sM = (rectDistP2[0]*rectAlongSideP2P3Unit[0]) + (rectDistP2[1]*rectAlongSideP2P3Unit[1]) + (rectDistP2[2]*rectAlongSideP2P3Unit[2]);
	sP = (rectDistP3[0]*rectAlongSideP2P3Unit[0]) + (rectDistP3[1]*rectAlongSideP2P3Unit[1]) + (rectDistP3[2]*rectAlongSideP2P3Unit[2]);

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

	// 3 //

	rM = rectDistP3Mag;
	rP = rectDistP0Mag;
	sM = (rectDistP3[0]*rectAlongSideP3P0Unit[0]) + (rectDistP3[1]*rectAlongSideP3P0Unit[1]) + (rectDistP3[2]*rectAlongSideP3P0Unit[2]);
	sP = (rectDistP0[0]*rectAlongSideP3P0Unit[0]) + (rectDistP0[1]*rectAlongSideP3P0Unit[1]) + (rectDistP0[2]*rectAlongSideP3P0Unit[2]);

	if ( (countCross==1) && (lineIndex==3) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			iL += ( t3 * (log( logArgNom )-log( LogArgTaylor(sM, dist) )) );
		}
	}

	if( lineIndex != 3 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		iL += ( t3 * (log(logArgNom)-log(logArgDenom)) );
	}

	return iL;
}

KThreeVector KElectrostaticRWGRectangleIntegrator::IqLField( const double* data, const double* P,
		const unsigned short countCross, const unsigned short lineIndex, const double dist ) const
{
	// computes first term of equation (74) in the case q = -1

    // corner points P0, P1, P2 and P3

    const double rectP0[3] = {
    		data[2], data[3], data[4] };

    const double rectP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const double rectP2[3] = {
    		data[2] + (data[0]*data[5]) + (data[1]*data[8]),
    		data[3] + (data[0]*data[6]) + (data[1]*data[9]),
			data[4] + (data[0]*data[7]) + (data[1]*data[10]) }; // = fP0 + fN1*fA + fN2*fB

    const double rectP3[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on rectangle surface

    double rectN3[3];
    rectN3[0] = data[6]*data[10] - data[7]*data[9];
    rectN3[1] = data[7]*data[8]  - data[5]*data[10];
    rectN3[2] = data[5]*data[9]  - data[6]*data[8];
    const double rectMagN3 = 1./sqrt( POW2(rectN3[0])+POW2(rectN3[1])+POW2(rectN3[2]) );
    rectN3[0] = rectN3[0]*rectMagN3;
    rectN3[1] = rectN3[1]*rectMagN3;
    rectN3[2] = rectN3[2]*rectMagN3;

    // side line unit vectors

    const double rectAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    const double rectAlongSideP1P2Unit[3] = {data[8], data[9], data[10]}; // = N2
    const double rectAlongSideP2P3Unit[3] = {-data[5], -data[6], -data[7]}; // = -N1
    const double rectAlongSideP3P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

    // outward pointing vector m, perpendicular to side lines

    const double m0[3] = {
    		(rectAlongSideP0P1Unit[1]*rectN3[2]) - (rectAlongSideP0P1Unit[2]*rectN3[1]),
			(rectAlongSideP0P1Unit[2]*rectN3[0]) - (rectAlongSideP0P1Unit[0]*rectN3[2]),
			(rectAlongSideP0P1Unit[0]*rectN3[1]) - (rectAlongSideP0P1Unit[1]*rectN3[0])
    };

    const double m1[3] = {
    		(rectAlongSideP1P2Unit[1]*rectN3[2]) - (rectAlongSideP1P2Unit[2]*rectN3[1]),
			(rectAlongSideP1P2Unit[2]*rectN3[0]) - (rectAlongSideP1P2Unit[0]*rectN3[2]),
			(rectAlongSideP1P2Unit[0]*rectN3[1]) - (rectAlongSideP1P2Unit[1]*rectN3[0])
    };

    const double m2[3] = {
    		(rectAlongSideP2P3Unit[1]*rectN3[2]) - (rectAlongSideP2P3Unit[2]*rectN3[1]),
			(rectAlongSideP2P3Unit[2]*rectN3[0]) - (rectAlongSideP2P3Unit[0]*rectN3[2]),
			(rectAlongSideP2P3Unit[0]*rectN3[1]) - (rectAlongSideP2P3Unit[1]*rectN3[0])
    };

    const double m3[3] = {
    		(rectAlongSideP3P0Unit[1]*rectN3[2]) - (rectAlongSideP3P0Unit[2]*rectN3[1]),
			(rectAlongSideP3P0Unit[2]*rectN3[0]) - (rectAlongSideP3P0Unit[0]*rectN3[2]),
			(rectAlongSideP3P0Unit[0]*rectN3[1]) - (rectAlongSideP3P0Unit[1]*rectN3[0])
    };

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const double rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const double rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const double rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const double rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };

   	const double rectDistP0Mag = sqrt( POW2(rectDistP0[0]) + POW2(rectDistP0[1]) + POW2(rectDistP0[2]) );
	const double rectDistP1Mag = sqrt( POW2(rectDistP1[0]) + POW2(rectDistP1[1]) + POW2(rectDistP1[2]) );
	const double rectDistP2Mag = sqrt( POW2(rectDistP2[0]) + POW2(rectDistP2[1]) + POW2(rectDistP2[2]) );
	const double rectDistP3Mag = sqrt( POW2(rectDistP3[0]) + POW2(rectDistP3[1]) + POW2(rectDistP3[2]) );

    // evaluation of line integral

    // 0 //

	double logArgNom, logArgDenom;
	double iL[3] = {0., 0., 0.};
	double tmpScalar;

	double rM = rectDistP0Mag;
	double rP = rectDistP1Mag;
	double sM = (rectDistP0[0]*rectAlongSideP0P1Unit[0]) + (rectDistP0[1]*rectAlongSideP0P1Unit[1]) + (rectDistP0[2]*rectAlongSideP0P1Unit[2]);
	double sP = (rectDistP1[0]*rectAlongSideP0P1Unit[0]) + (rectDistP1[1]*rectAlongSideP0P1Unit[1]) + (rectDistP1[2]*rectAlongSideP0P1Unit[2]);

	if ( (countCross==1) && (lineIndex==0) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = (log( logArgNom )-log( LogArgTaylor(sM, dist) ));
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

		tmpScalar = (log( logArgNom )-log( logArgDenom ));
		iL[0] += ( m0[0] * tmpScalar );
		iL[1] += ( m0[1] * tmpScalar );
		iL[2] += ( m0[2] * tmpScalar );
	}

	// 1 //

	rM = rectDistP1Mag;
	rP = rectDistP2Mag;
	sM = (rectDistP1[0]*rectAlongSideP1P2Unit[0]) + (rectDistP1[1]*rectAlongSideP1P2Unit[1]) + (rectDistP1[2]*rectAlongSideP1P2Unit[2]);
	sP = (rectDistP2[0]*rectAlongSideP1P2Unit[0]) + (rectDistP2[1]*rectAlongSideP1P2Unit[1]) + (rectDistP2[2]*rectAlongSideP1P2Unit[2]);

	if ( (countCross==1) && (lineIndex==1) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = (log( logArgNom )-log( LogArgTaylor(sM, dist) ));
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
		tmpScalar = (log( logArgNom )-log( logArgDenom ));
		iL[0] += ( m1[0] * tmpScalar );
		iL[1] += ( m1[1] * tmpScalar );
		iL[2] += ( m1[2] * tmpScalar );
	}

	// 2 //

	rM = rectDistP2Mag;
	rP = rectDistP3Mag;
	sM = (rectDistP2[0]*rectAlongSideP2P3Unit[0]) + (rectDistP2[1]*rectAlongSideP2P3Unit[1]) + (rectDistP2[2]*rectAlongSideP2P3Unit[2]);
	sP = (rectDistP3[0]*rectAlongSideP2P3Unit[0]) + (rectDistP3[1]*rectAlongSideP2P3Unit[1]) + (rectDistP3[2]*rectAlongSideP2P3Unit[2]);

	if ( (countCross==1) && (lineIndex==2) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = (log( logArgNom )-log( LogArgTaylor(sM, dist) ));
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
		tmpScalar = (log( logArgNom )-log( logArgDenom ));
		iL[0] += ( m2[0] * tmpScalar );
		iL[1] += ( m2[1] * tmpScalar );
		iL[2] += ( m2[2] * tmpScalar );
	}

	// 3 //

	rM = rectDistP3Mag;
	rP = rectDistP0Mag;
	sM = (rectDistP3[0]*rectAlongSideP3P0Unit[0]) + (rectDistP3[1]*rectAlongSideP3P0Unit[1]) + (rectDistP3[2]*rectAlongSideP3P0Unit[2]);
	sP = (rectDistP0[0]*rectAlongSideP3P0Unit[0]) + (rectDistP0[1]*rectAlongSideP3P0Unit[1]) + (rectDistP0[2]*rectAlongSideP3P0Unit[2]);

	if ( (countCross==1) && (lineIndex==3) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = (log( logArgNom )-log( LogArgTaylor(sM, dist) ));
			iL[0] += ( m3[0] * tmpScalar );
			iL[1] += ( m3[1] * tmpScalar );
			iL[2] += ( m3[2] * tmpScalar );
		}
	}

	if( lineIndex != 3 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		tmpScalar = (log( logArgNom )-log( logArgDenom ));
		iL[0] += ( m3[0] * tmpScalar );
		iL[1] += ( m3[1] * tmpScalar );
		iL[2] += ( m3[2] * tmpScalar );
	}

	return KThreeVector( iL[0], iL[1], iL[2] );
}

std::pair<KThreeVector, double> KElectrostaticRWGRectangleIntegrator::IqLFieldAndPotential( const double* data, const double* P,
		const unsigned short countCross, const unsigned short lineIndex, const double dist ) const
{
    // corner points P0, P1, P2 and P3

    const double rectP0[3] = {
    		data[2], data[3], data[4] };

    const double rectP1[3] = {
    		data[2] + (data[0]*data[5]),
    		data[3] + (data[0]*data[6]),
			data[4] + (data[0]*data[7]) }; // = fP0 + fN1*fA

    const double rectP2[3] = {
    		data[2] + (data[0]*data[5]) + (data[1]*data[8]),
    		data[3] + (data[0]*data[6]) + (data[1]*data[9]),
			data[4] + (data[0]*data[7]) + (data[1]*data[10]) }; // = fP0 + fN1*fA + fN2*fB

    const double rectP3[3] = {
    		data[2] + (data[1]*data[8]),
    		data[3] + (data[1]*data[9]),
			data[4] + (data[1]*data[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on rectangle surface

    double rectN3[3];
    rectN3[0] = data[6]*data[10] - data[7]*data[9];
    rectN3[1] = data[7]*data[8]  - data[5]*data[10];
    rectN3[2] = data[5]*data[9]  - data[6]*data[8];
    const double rectMagN3 = 1./sqrt( POW2(rectN3[0])+POW2(rectN3[1])+POW2(rectN3[2]) );
    rectN3[0] = rectN3[0]*rectMagN3;
    rectN3[1] = rectN3[1]*rectMagN3;
    rectN3[2] = rectN3[2]*rectMagN3;

    // side line unit vectors

    const double rectAlongSideP0P1Unit[3] = {data[5], data[6], data[7]}; // = N1
    const double rectAlongSideP1P2Unit[3] = {data[8], data[9], data[10]}; // = N2
    const double rectAlongSideP2P3Unit[3] = {-data[5], -data[6], -data[7]}; // = -N1
    const double rectAlongSideP3P0Unit[3] = {-data[8], -data[9], -data[10]}; // = -N2

    // center point of each side to field point

    const double e0[3] = {
    		(0.5*data[0]*data[5]) + rectP0[0],
			(0.5*data[0]*data[6]) + rectP0[1],
			(0.5*data[0]*data[7]) + rectP0[2] };

    const double e1[3] = {
    		(0.5*data[1]*data[8]) + rectP1[0],
			(0.5*data[1]*data[9]) + rectP1[1],
			(0.5*data[1]*data[10]) + rectP1[2] };

    const double e2[3] = {
    		(-0.5*data[0]*data[5]) + rectP2[0],
			(-0.5*data[0]*data[6]) + rectP2[1],
			(-0.5*data[0]*data[7]) + rectP2[2] };

    const double e3[3] = {
    		(-0.5*data[1]*data[8]) + rectP3[0],
			(-0.5*data[1]*data[9]) + rectP3[1],
			(-0.5*data[1]*data[10]) + rectP3[2] };

    // outward pointing vector m, perpendicular to side lines

    const double m0[3] = {
    		(rectAlongSideP0P1Unit[1]*rectN3[2]) - (rectAlongSideP0P1Unit[2]*rectN3[1]),
			(rectAlongSideP0P1Unit[2]*rectN3[0]) - (rectAlongSideP0P1Unit[0]*rectN3[2]),
			(rectAlongSideP0P1Unit[0]*rectN3[1]) - (rectAlongSideP0P1Unit[1]*rectN3[0])
    };

    const double m1[3] = {
    		(rectAlongSideP1P2Unit[1]*rectN3[2]) - (rectAlongSideP1P2Unit[2]*rectN3[1]),
			(rectAlongSideP1P2Unit[2]*rectN3[0]) - (rectAlongSideP1P2Unit[0]*rectN3[2]),
			(rectAlongSideP1P2Unit[0]*rectN3[1]) - (rectAlongSideP1P2Unit[1]*rectN3[0])
    };

    const double m2[3] = {
    		(rectAlongSideP2P3Unit[1]*rectN3[2]) - (rectAlongSideP2P3Unit[2]*rectN3[1]),
			(rectAlongSideP2P3Unit[2]*rectN3[0]) - (rectAlongSideP2P3Unit[0]*rectN3[2]),
			(rectAlongSideP2P3Unit[0]*rectN3[1]) - (rectAlongSideP2P3Unit[1]*rectN3[0])
    };

    const double m3[3] = {
    		(rectAlongSideP3P0Unit[1]*rectN3[2]) - (rectAlongSideP3P0Unit[2]*rectN3[1]),
			(rectAlongSideP3P0Unit[2]*rectN3[0]) - (rectAlongSideP3P0Unit[0]*rectN3[2]),
			(rectAlongSideP3P0Unit[0]*rectN3[1]) - (rectAlongSideP3P0Unit[1]*rectN3[0])
    };

    // size t

    const double t0 = (m0[0]*(P[0]-e0[0])) + (m0[1]*(P[1]-e0[1])) + (m0[2]*(P[2]-e0[2]));
    const double t1 = (m1[0]*(P[0]-e1[0])) + (m1[1]*(P[1]-e1[1])) + (m1[2]*(P[2]-e1[2]));
    const double t2 = (m2[0]*(P[0]-e2[0])) + (m2[1]*(P[1]-e2[1])) + (m2[2]*(P[2]-e2[2]));
    const double t3 = (m3[0]*(P[0]-e3[0])) + (m3[1]*(P[1]-e3[1])) + (m3[2]*(P[2]-e3[2]));

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const double rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const double rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const double rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const double rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };

   	const double rectDistP0Mag = sqrt( POW2(rectDistP0[0]) + POW2(rectDistP0[1]) + POW2(rectDistP0[2]) );
	const double rectDistP1Mag = sqrt( POW2(rectDistP1[0]) + POW2(rectDistP1[1]) + POW2(rectDistP1[2]) );
	const double rectDistP2Mag = sqrt( POW2(rectDistP2[0]) + POW2(rectDistP2[1]) + POW2(rectDistP2[2]) );
	const double rectDistP3Mag = sqrt( POW2(rectDistP3[0]) + POW2(rectDistP3[1]) + POW2(rectDistP3[2]) );

    // evaluation of line integral

    // 0 //

	double logArgNom, logArgDenom;
	double iLPhi = 0.;
	double iLField[3] = {0., 0., 0.};
	double tmpScalar;

	double rM = rectDistP0Mag;
	double rP = rectDistP1Mag;
	double sM = (rectDistP0[0]*rectAlongSideP0P1Unit[0]) + (rectDistP0[1]*rectAlongSideP0P1Unit[1]) + (rectDistP0[2]*rectAlongSideP0P1Unit[2]);
	double sP = (rectDistP1[0]*rectAlongSideP0P1Unit[0]) + (rectDistP1[1]*rectAlongSideP0P1Unit[1]) + (rectDistP1[2]*rectAlongSideP0P1Unit[2]);

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

	rM = rectDistP1Mag;
	rP = rectDistP2Mag;
	sM = (rectDistP1[0]*rectAlongSideP1P2Unit[0]) + (rectDistP1[1]*rectAlongSideP1P2Unit[1]) + (rectDistP1[2]*rectAlongSideP1P2Unit[2]);
	sP = (rectDistP2[0]*rectAlongSideP1P2Unit[0]) + (rectDistP2[1]*rectAlongSideP1P2Unit[1]) + (rectDistP2[2]*rectAlongSideP1P2Unit[2]);

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

	rM = rectDistP2Mag;
	rP = rectDistP3Mag;
	sM = (rectDistP2[0]*rectAlongSideP2P3Unit[0]) + (rectDistP2[1]*rectAlongSideP2P3Unit[1]) + (rectDistP2[2]*rectAlongSideP2P3Unit[2]);
	sP = (rectDistP3[0]*rectAlongSideP2P3Unit[0]) + (rectDistP3[1]*rectAlongSideP2P3Unit[1]) + (rectDistP3[2]*rectAlongSideP2P3Unit[2]);

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

	// 3 //

	rM = rectDistP3Mag;
	rP = rectDistP0Mag;
	sM = (rectDistP3[0]*rectAlongSideP3P0Unit[0]) + (rectDistP3[1]*rectAlongSideP3P0Unit[1]) + (rectDistP3[2]*rectAlongSideP3P0Unit[2]);
	sP = (rectDistP0[0]*rectAlongSideP3P0Unit[0]) + (rectDistP0[1]*rectAlongSideP3P0Unit[1]) + (rectDistP0[2]*rectAlongSideP3P0Unit[2]);

	if ( (countCross==1) && (lineIndex==3) ) {
		if( fabs(dist/sM) < fLogArgQuotient ) {
			logArgNom = (rP+sP);
			tmpScalar = (log( logArgNom )-log( LogArgTaylor(sM, dist) ));
			iLField[0] += ( m3[0] * tmpScalar );
			iLField[1] += ( m3[1] * tmpScalar );
			iLField[2] += ( m3[2] * tmpScalar );
			iLPhi += ( t3 * tmpScalar );
		}
	}

	if( lineIndex != 3 ) {
		if( (rM+sM) > (rP-sP) ) {
			logArgNom = (rP+sP);
			logArgDenom = (rM+sM);
		}
		else {
			logArgNom = (rM-sM);
			logArgDenom = (rP-sP);
		}
		tmpScalar = (log( logArgNom )-log( logArgDenom ));
		iLField[0] += ( m3[0] * tmpScalar );
		iLField[1] += ( m3[1] * tmpScalar );
		iLField[2] += ( m3[2] * tmpScalar );
		iLPhi += ( t3 * tmpScalar );
	}

	return std::make_pair( iLField, iLPhi );
}

double KElectrostaticRWGRectangleIntegrator::Potential( const KRectangle* source, const KPosition& P ) const
{
	// save rectangle data into double array

	const double rectData[11] = {
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

	// corner points P0, P1, P2 and P3

    const double rectP0[3] = {
    		rectData[2], rectData[3], rectData[4] };

    const double rectP1[3] = {
    		rectData[2] + (rectData[0]*rectData[5]),
			rectData[3] + (rectData[0]*rectData[6]),
			rectData[4] + (rectData[0]*rectData[7]) }; // = fP0 + fN1*fA

    const double rectP2[3] = {
    		rectData[2] + (rectData[0]*rectData[5]) + (rectData[1]*rectData[8]),
			rectData[3] + (rectData[0]*rectData[6]) + (rectData[1]*rectData[9]),
			rectData[4] + (rectData[0]*rectData[7]) + (rectData[1]*rectData[10]) }; // = fP0 + fN1*fA + fN2*fB

    const double rectP3[3] = {
    		rectData[2] + (rectData[1]*rectData[8]),
			rectData[3] + (rectData[1]*rectData[9]),
			rectData[4] + (rectData[1]*rectData[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on rectangle surface

    double rectN3[3];
    rectN3[0] = rectData[6]*rectData[10] - rectData[7]*rectData[9];
    rectN3[1] = rectData[7]*rectData[8]  - rectData[5]*rectData[10];
    rectN3[2] = rectData[5]*rectData[9]  - rectData[6]*rectData[8];
    const double rectMagN3 = 1./sqrt( POW2(rectN3[0])+POW2(rectN3[1])+POW2(rectN3[2]) );
    rectN3[0] = rectN3[0]*rectMagN3;
    rectN3[1] = rectN3[1]*rectMagN3;
    rectN3[2] = rectN3[2]*rectMagN3;

    // rectangle centroid

    const double rectCenter[3] = {
    		rectData[2] + 0.5*(rectData[0]*rectData[5]) + 0.5*(rectData[1]*rectData[8]),
			rectData[3] + 0.5*(rectData[0]*rectData[6]) + 0.5*(rectData[1]*rectData[9]),
			rectData[4] + 0.5*(rectData[0]*rectData[7]) + 0.5*(rectData[1]*rectData[10])};

    // side line vectors

    const double rectAlongSideP0P1[3] = {
    		rectData[0]*rectData[5],
			rectData[0]*rectData[6],
			rectData[0]*rectData[7] }; // = A * N1
    const double rectAlongSideP1P2[3] = {
    		rectData[1]*rectData[8],
			rectData[1]*rectData[9],
			rectData[1]*rectData[10] }; // = B * N2
    const double rectAlongSideP2P3[3] = {
    		-rectData[0]*rectData[5],
			-rectData[0]*rectData[6],
			-rectData[0]*rectData[7] }; // = -A * N1
    const double rectAlongSideP3P0[3] = {
    		-rectData[1]*rectData[8],
			-rectData[1]*rectData[9],
			-rectData[1]*rectData[10] }; // = -B * N2

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const double rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const double rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const double rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const double rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };

	// check if field point is at rectangle edge or at side line

    double distToLine = 0.;
    double distToLineMin = 0.;

    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on rectangle edge (= 2) or point is on side line (= 1)
    double lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    double tmpVector[3];
    double tmpScalar;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./rectData[0];

    // compute cross product

    tmpVector[0] = (rectAlongSideP0P1[1]*rectDistP0[2]) - (rectAlongSideP0P1[2]*rectDistP0[1]);
    tmpVector[1] = (rectAlongSideP0P1[2]*rectDistP0[0]) - (rectAlongSideP0P1[0]*rectDistP0[2]);
    tmpVector[2] = (rectAlongSideP0P1[0]*rectDistP0[1]) - (rectAlongSideP0P1[1]*rectDistP0[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0
    lineLambda = ((-rectDistP0[0]*rectAlongSideP0P1[0]) + (-rectDistP0[1]*rectAlongSideP0P1[1]) + (-rectDistP0[2]*rectAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
    	if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./rectData[1];

    // compute cross product

    tmpVector[0] = (rectAlongSideP1P2[1]*rectDistP1[2]) - (rectAlongSideP1P2[2]*rectDistP1[1]);
    tmpVector[1] = (rectAlongSideP1P2[2]*rectDistP1[0]) - (rectAlongSideP1P2[0]*rectDistP1[2]);
    tmpVector[2] = (rectAlongSideP1P2[0]*rectDistP1[1]) - (rectAlongSideP1P2[1]*rectDistP1[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP1[0]*rectAlongSideP1P2[0]) + (-rectDistP1[1]*rectAlongSideP1P2[1]) + (-rectDistP1[2]*rectAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P3 side line

    tmpScalar = 1./rectData[0];

    tmpVector[0] = (rectAlongSideP2P3[1]*rectDistP2[2]) - (rectAlongSideP2P3[2]*rectDistP2[1]);
    tmpVector[1] = (rectAlongSideP2P3[2]*rectDistP2[0]) - (rectAlongSideP2P3[0]*rectDistP2[2]);
    tmpVector[2] = (rectAlongSideP2P3[0]*rectDistP2[1]) - (rectAlongSideP2P3[1]*rectDistP2[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP2[0]*rectAlongSideP2P3[0]) + (-rectDistP2[1]*rectAlongSideP2P3[1]) + (-rectDistP2[2]*rectAlongSideP2P3[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // 3 - check distances to P3P0 side line

    tmpScalar = 1./rectData[1];

    // compute cross product

    tmpVector[0] = (rectAlongSideP3P0[1]*rectDistP3[2]) - (rectAlongSideP3P0[2]*rectDistP3[1]);
    tmpVector[1] = (rectAlongSideP3P0[2]*rectDistP3[0]) - (rectAlongSideP3P0[0]*rectDistP3[2]);
    tmpVector[2] = (rectAlongSideP3P0[0]*rectDistP3[1]) - (rectAlongSideP3P0[1]*rectDistP3[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for rectDistP3
    lineLambda = ((-rectDistP3[0]*rectAlongSideP3P0[0]) + (-rectDistP3[1]*rectAlongSideP3P0[1]) + (-rectDistP3[2]*rectAlongSideP3P0[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 3;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length CORRECTIONREN3 in positive and negative N3 direction

    if( correctionCounter == 2 ) {
       	const double upEps[3] = {
       			P[0] + fDistanceCorrectionN3*rectN3[0],
				P[1] + fDistanceCorrectionN3*rectN3[1],
				P[2] + fDistanceCorrectionN3*rectN3[2]
       	};

    	const double downEps[3] = {
    			P[0] - fDistanceCorrectionN3*rectN3[0],
				P[1] - fDistanceCorrectionN3*rectN3[1],
				P[2] - fDistanceCorrectionN3*rectN3[2]
    	};

    	// compute IqS term

        const double hUp = ( rectN3[0] * (upEps[0]-rectCenter[0]) )
				+ ( rectN3[1] * (upEps[1]-rectCenter[1]) )
				+ ( rectN3[2] * (upEps[2]-rectCenter[2]) );

        const double solidAngleUp = solidAngle.SolidAngleRectangleAsArray( rectData, upEps );

        const double hDown = ( rectN3[0] * (downEps[0]-rectCenter[0]) )
				+ ( rectN3[1] * (downEps[1]-rectCenter[1]) )
				+ ( rectN3[2] * (downEps[2]-rectCenter[2]) );

        const double solidAngleDown = solidAngle.SolidAngleRectangleAsArray( rectData, downEps );

    	// compute IqL

    	const double IqLUp = IqLPotential( rectData, upEps, 9, 9, 9 ); /* no line correction */

    	const double IqLDown = IqLPotential( rectData, downEps, 9, 9, 9 ); /* no line correction */

    	const double finalResult = 0.5*((-hUp*solidAngleUp - IqLUp) + (-hDown*solidAngleDown - IqLDown));

    	return finalResult*KEMConstants::OneOverFourPiEps0;
    }

    const double h = ( rectN3[0] * (P[0]-rectCenter[0]) )
			+ ( rectN3[1] * (P[1]-rectCenter[1]) )
			+ ( rectN3[2] * (P[2]-rectCenter[2]) );

    double fieldPoint[3] = { P[0], P[1], P[2] };

    const double rectSolidAngle = solidAngle.SolidAngleRectangleAsArray( rectData, fieldPoint );

	const double finalResult = (-h*rectSolidAngle) - IqLPotential( rectData, fieldPoint, correctionCounter, correctionLineIndex, distToLineMin );

    return finalResult*KEMConstants::OneOverFourPiEps0;
}

KThreeVector KElectrostaticRWGRectangleIntegrator::ElectricField( const KRectangle* source, const KPosition& P ) const
{
	// save rectangle data into double array

	const double rectData[11] = {
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

	// corner points P0, P1, P2 and P3

    const double rectP0[3] = {
    		rectData[2], rectData[3], rectData[4] };

    const double rectP1[3] = {
    		rectData[2] + (rectData[0]*rectData[5]),
			rectData[3] + (rectData[0]*rectData[6]),
			rectData[4] + (rectData[0]*rectData[7]) }; // = fP0 + fN1*fA

    const double rectP2[3] = {
    		rectData[2] + (rectData[0]*rectData[5]) + (rectData[1]*rectData[8]),
			rectData[3] + (rectData[0]*rectData[6]) + (rectData[1]*rectData[9]),
			rectData[4] + (rectData[0]*rectData[7]) + (rectData[1]*rectData[10]) }; // = fP0 + fN1*fA + fN2*fB

    const double rectP3[3] = {
    		rectData[2] + (rectData[1]*rectData[8]),
			rectData[3] + (rectData[1]*rectData[9]),
			rectData[4] + (rectData[1]*rectData[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on rectangle surface

    double rectN3[3];
    rectN3[0] = rectData[6]*rectData[10] - rectData[7]*rectData[9];
    rectN3[1] = rectData[7]*rectData[8]  - rectData[5]*rectData[10];
    rectN3[2] = rectData[5]*rectData[9]  - rectData[6]*rectData[8];
    const double rectMagN3 = 1./sqrt( POW2(rectN3[0])+POW2(rectN3[1])+POW2(rectN3[2]) );
    rectN3[0] = rectN3[0]*rectMagN3;
    rectN3[1] = rectN3[1]*rectMagN3;
    rectN3[2] = rectN3[2]*rectMagN3;

    // side line vectors

    const double rectAlongSideP0P1[3] = {
    		rectData[0]*rectData[5],
			rectData[0]*rectData[6],
			rectData[0]*rectData[7] }; // = A * N1
    const double rectAlongSideP1P2[3] = {
    		rectData[1]*rectData[8],
			rectData[1]*rectData[9],
			rectData[1]*rectData[10] }; // = B * N2
    const double rectAlongSideP2P3[3] = {
    		-rectData[0]*rectData[5],
			-rectData[0]*rectData[6],
			-rectData[0]*rectData[7] }; // = -A * N1
    const double rectAlongSideP3P0[3] = {
    		-rectData[1]*rectData[8],
			-rectData[1]*rectData[9],
			-rectData[1]*rectData[10] }; // = -B * N2

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const double rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const double rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const double rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const double rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };

	// check if field point is at rectangle edge or at side line

    double distToLine = 0.;
    double distToLineMin = 0.;

    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on rectangle edge (= 2) or point is on side line (= 1)
    double lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    double tmpVector[3];
    double tmpScalar;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./rectData[0];

    // compute cross product

    tmpVector[0] = (rectAlongSideP0P1[1]*rectDistP0[2]) - (rectAlongSideP0P1[2]*rectDistP0[1]);
    tmpVector[1] = (rectAlongSideP0P1[2]*rectDistP0[0]) - (rectAlongSideP0P1[0]*rectDistP0[2]);
    tmpVector[2] = (rectAlongSideP0P1[0]*rectDistP0[1]) - (rectAlongSideP0P1[1]*rectDistP0[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array triDistP0
    lineLambda = ((-rectDistP0[0]*rectAlongSideP0P1[0]) + (-rectDistP0[1]*rectAlongSideP0P1[1]) + (-rectDistP0[2]*rectAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./rectData[1];

    // compute cross product

    tmpVector[0] = (rectAlongSideP1P2[1]*rectDistP1[2]) - (rectAlongSideP1P2[2]*rectDistP1[1]);
    tmpVector[1] = (rectAlongSideP1P2[2]*rectDistP1[0]) - (rectAlongSideP1P2[0]*rectDistP1[2]);
    tmpVector[2] = (rectAlongSideP1P2[0]*rectDistP1[1]) - (rectAlongSideP1P2[1]*rectDistP1[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP1[0]*rectAlongSideP1P2[0]) + (-rectDistP1[1]*rectAlongSideP1P2[1]) + (-rectDistP1[2]*rectAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P3 side line

    tmpScalar = 1./rectData[0];

    tmpVector[0] = (rectAlongSideP2P3[1]*rectDistP2[2]) - (rectAlongSideP2P3[2]*rectDistP2[1]);
    tmpVector[1] = (rectAlongSideP2P3[2]*rectDistP2[0]) - (rectAlongSideP2P3[0]*rectDistP2[2]);
    tmpVector[2] = (rectAlongSideP2P3[0]*rectDistP2[1]) - (rectAlongSideP2P3[1]*rectDistP2[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP2[0]*rectAlongSideP2P3[0]) + (-rectDistP2[1]*rectAlongSideP2P3[1]) + (-rectDistP2[2]*rectAlongSideP2P3[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // 3 - check distances to P3P0 side line

    tmpScalar = 1./rectData[1];

    // compute cross product

    tmpVector[0] = (rectAlongSideP3P0[1]*rectDistP3[2]) - (rectAlongSideP3P0[2]*rectDistP3[1]);
    tmpVector[1] = (rectAlongSideP3P0[2]*rectDistP3[0]) - (rectAlongSideP3P0[0]*rectDistP3[2]);
    tmpVector[2] = (rectAlongSideP3P0[0]*rectDistP3[1]) - (rectAlongSideP3P0[1]*rectDistP3[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for rectDistP3
    lineLambda = ((-rectDistP3[0]*rectAlongSideP3P0[0]) + (-rectDistP3[1]*rectAlongSideP3P0[1]) + (-rectDistP3[2]*rectAlongSideP3P0[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 3;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length fDistanceCorrectionN3 in positive and negative N3 direction

    if( correctionCounter == 2 ) {
    	const double upEps[3] = {
    			P[0] + fDistanceCorrectionN3*rectN3[0],
				P[1] + fDistanceCorrectionN3*rectN3[1],
				P[2] + fDistanceCorrectionN3*rectN3[2]
    	};
    	const double downEps[3] = {
    			P[0] - fDistanceCorrectionN3*rectN3[0],
				P[1] - fDistanceCorrectionN3*rectN3[1],
				P[2] - fDistanceCorrectionN3*rectN3[2]
    	};

    	// compute IqS term

        const double solidAngleUp
			= solidAngle.SolidAngleRectangleAsArray( rectData, upEps );

        const double solidAngleDown
			= solidAngle.SolidAngleRectangleAsArray( rectData, downEps );

    	// compute IqL

    	const KThreeVector IqLUp
			= IqLField( rectData, upEps, 9, 9, 9 ); /* no line correction */
    	const KThreeVector IqLDown
			= IqLField( rectData, downEps, 9, 9, 9 ); /* no line correction */

    	const KThreeVector finalResult(
    			(rectN3[0]*solidAngleUp + IqLUp[0]) + (rectN3[0]*solidAngleDown + IqLDown[0]),
				(rectN3[1]*solidAngleUp + IqLUp[1]) + (rectN3[1]*solidAngleDown + IqLDown[1]),
				(rectN3[2]*solidAngleUp + IqLUp[2]) + (rectN3[2]*solidAngleDown + IqLDown[2]) );

    	return (0.5*finalResult)*KEMConstants::OneOverFourPiEps0;
    }

    double fieldPoint[3] = { P[0], P[1], P[2] };

    const double rectSolidAngle = solidAngle.SolidAngleRectangleAsArray( rectData, fieldPoint );

    const KThreeVector IqLEField
		= IqLField( rectData, fieldPoint, correctionCounter, correctionLineIndex, distToLineMin );

    const KThreeVector finalResult (
    		(rectN3[0]*rectSolidAngle + IqLEField.X()),
    		(rectN3[1]*rectSolidAngle + IqLEField.Y()),
			(rectN3[2]*rectSolidAngle + IqLEField.Z()) );

    return finalResult*KEMConstants::OneOverFourPiEps0;
}

std::pair<KThreeVector, double> KElectrostaticRWGRectangleIntegrator::ElectricFieldAndPotential( const KRectangle* source, const KPosition& P ) const
{
	// save rectangle data into double array

	const double rectData[11] = {
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

	// corner points P0, P1, P2 and P3

    const double rectP0[3] = {
    		rectData[2], rectData[3], rectData[4] };

    const double rectP1[3] = {
    		rectData[2] + (rectData[0]*rectData[5]),
			rectData[3] + (rectData[0]*rectData[6]),
			rectData[4] + (rectData[0]*rectData[7]) }; // = fP0 + fN1*fA

    const double rectP2[3] = {
    		rectData[2] + (rectData[0]*rectData[5]) + (rectData[1]*rectData[8]),
			rectData[3] + (rectData[0]*rectData[6]) + (rectData[1]*rectData[9]),
			rectData[4] + (rectData[0]*rectData[7]) + (rectData[1]*rectData[10]) }; // = fP0 + fN1*fA + fN2*fB

    const double rectP3[3] = {
    		rectData[2] + (rectData[1]*rectData[8]),
			rectData[3] + (rectData[1]*rectData[9]),
			rectData[4] + (rectData[1]*rectData[10]) }; // = fP0 + fN2*fB

	// get perpendicular normal vector n3 on rectangle surface

    double rectN3[3];
    rectN3[0] = rectData[6]*rectData[10] - rectData[7]*rectData[9];
    rectN3[1] = rectData[7]*rectData[8]  - rectData[5]*rectData[10];
    rectN3[2] = rectData[5]*rectData[9]  - rectData[6]*rectData[8];
    const double rectMagN3 = 1./sqrt( POW2(rectN3[0])+POW2(rectN3[1])+POW2(rectN3[2]) );
    rectN3[0] = rectN3[0]*rectMagN3;
    rectN3[1] = rectN3[1]*rectMagN3;
    rectN3[2] = rectN3[2]*rectMagN3;

    // rectangle centroid

    const double rectCenter[3] = {
    		rectData[2] + 0.5*(rectData[0]*rectData[5]) + 0.5*(rectData[1]*rectData[8]),
			rectData[3] + 0.5*(rectData[0]*rectData[6]) + 0.5*(rectData[1]*rectData[9]),
			rectData[4] + 0.5*(rectData[0]*rectData[7]) + 0.5*(rectData[1]*rectData[10])};

    // side line vectors

    const double rectAlongSideP0P1[3] = {
    		rectData[0]*rectData[5],
			rectData[0]*rectData[6],
			rectData[0]*rectData[7] }; // = A * N1
    const double rectAlongSideP1P2[3] = {
    		rectData[1]*rectData[8],
			rectData[1]*rectData[9],
			rectData[1]*rectData[10] }; // = B * N2
    const double rectAlongSideP2P3[3] = {
    		-rectData[0]*rectData[5],
			-rectData[0]*rectData[6],
			-rectData[0]*rectData[7] }; // = -A * N1
    const double rectAlongSideP3P0[3] = {
    		-rectData[1]*rectData[8],
			-rectData[1]*rectData[9],
			-rectData[1]*rectData[10] }; // = -B * N2

    // distance between rectangle vertex points and field points in positive rotation order
    // pointing to the rectangle vertex point

    const double rectDistP0[3] = {
    		rectP0[0] - P[0],
			rectP0[1] - P[1],
			rectP0[2] - P[2] };

    const double rectDistP1[3] = {
    		rectP1[0] - P[0],
			rectP1[1] - P[1],
			rectP1[2] - P[2] };

    const double rectDistP2[3] = {
    		rectP2[0] - P[0],
			rectP2[1] - P[1],
			rectP2[2] - P[2] };

    const double rectDistP3[3] = {
    		rectP3[0] - P[0],
			rectP3[1] - P[1],
			rectP3[2] - P[2] };

	// check if field point is at rectangle edge or at side line

    double distToLine = 0.;
    double distToLineMin = 0.;

    unsigned short correctionLineIndex = 99; // index of crossing line
    unsigned short correctionCounter = 0; // point is on rectangle edge (= 2) or point is on side line (= 1)
    double lineLambda = -1.; /* parameter for distance vector */

    // auxiliary values for distance check

    double tmpVector[3];
    double tmpScalar;

    // 0 - check distances to P0P1 side line

    tmpScalar = 1./rectData[0];

    // compute cross product

    tmpVector[0] = (rectAlongSideP0P1[1]*rectDistP0[2]) - (rectAlongSideP0P1[2]*rectDistP0[1]);
    tmpVector[1] = (rectAlongSideP0P1[2]*rectDistP0[0]) - (rectAlongSideP0P1[0]*rectDistP0[2]);
    tmpVector[2] = (rectAlongSideP0P1[0]*rectDistP0[1]) - (rectAlongSideP0P1[1]*rectDistP0[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 in order to use array rectDistP0
    lineLambda = ((-rectDistP0[0]*rectAlongSideP0P1[0]) + (-rectDistP0[1]*rectAlongSideP0P1[1]) + (-rectDistP0[2]*rectAlongSideP0P1[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 0;
    	} /* lambda */
    } /* distance */

    // 1 - check distances to P1P2 side line

    tmpScalar = 1./rectData[1];

    // compute cross product

    tmpVector[0] = (rectAlongSideP1P2[1]*rectDistP1[2]) - (rectAlongSideP1P2[2]*rectDistP1[1]);
    tmpVector[1] = (rectAlongSideP1P2[2]*rectDistP1[0]) - (rectAlongSideP1P2[0]*rectDistP1[2]);
    tmpVector[2] = (rectAlongSideP1P2[0]*rectDistP1[1]) - (rectAlongSideP1P2[1]*rectDistP1[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP1[0]*rectAlongSideP1P2[0]) + (-rectDistP1[1]*rectAlongSideP1P2[1]) + (-rectDistP1[2]*rectAlongSideP1P2[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 1;
    	} /* lambda */
    } /* distance */

    // 2 - check distances to P2P3 side line

    tmpScalar = 1./rectData[0];

    tmpVector[0] = (rectAlongSideP2P3[1]*rectDistP2[2]) - (rectAlongSideP2P3[2]*rectDistP2[1]);
    tmpVector[1] = (rectAlongSideP2P3[2]*rectDistP2[0]) - (rectAlongSideP2P3[0]*rectDistP2[2]);
    tmpVector[2] = (rectAlongSideP2P3[0]*rectDistP2[1]) - (rectAlongSideP2P3[1]*rectDistP2[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for direction rectDistP1 vector
    lineLambda = ((-rectDistP2[0]*rectAlongSideP2P3[0]) + (-rectDistP2[1]*rectAlongSideP2P3[1]) + (-rectDistP2[2]*rectAlongSideP2P3[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 2;
    	} /* lambda */
    } /* distance */

    // 3 - check distances to P3P0 side line

    tmpScalar = 1./rectData[1];

    // compute cross product

    tmpVector[0] = (rectAlongSideP3P0[1]*rectDistP3[2]) - (rectAlongSideP3P0[2]*rectDistP3[1]);
    tmpVector[1] = (rectAlongSideP3P0[2]*rectDistP3[0]) - (rectAlongSideP3P0[0]*rectDistP3[2]);
    tmpVector[2] = (rectAlongSideP3P0[0]*rectDistP3[1]) - (rectAlongSideP3P0[1]*rectDistP3[0]);

    distToLine = sqrt( POW2(tmpVector[0]) + POW2(tmpVector[1]) + POW2(tmpVector[2]) ) * tmpScalar;

    // factor -1 for rectDistP3
    lineLambda = ((-rectDistP3[0]*rectAlongSideP3P0[0]) + (-rectDistP3[1]*rectAlongSideP3P0[1]) + (-rectDistP3[2]*rectAlongSideP3P0[2])) * POW2( tmpScalar );

    if( distToLine < fMinDistanceToSideLine ) {
        if( lineLambda >=-fToleranceLambda && lineLambda <=(1.+fToleranceLambda) ) {
    		distToLineMin = distToLine;
    		correctionCounter++;
    		correctionLineIndex = 3;
    	} /* lambda */
    } /* distance */

    // if point is on edge, the field point will be moved with length CORRECTIONREN3 in positive and negative N3 direction

    if( correctionCounter == 2 ) {
       	const double upEps[3] = {
       			P[0] + fDistanceCorrectionN3*rectN3[0],
				P[1] + fDistanceCorrectionN3*rectN3[1],
				P[2] + fDistanceCorrectionN3*rectN3[2]
       	};

    	const double downEps[3] = {
    			P[0] - fDistanceCorrectionN3*rectN3[0],
				P[1] - fDistanceCorrectionN3*rectN3[1],
				P[2] - fDistanceCorrectionN3*rectN3[2]
    	};

    	// compute IqS term

        const double hUp = ( rectN3[0] * (upEps[0]-rectCenter[0]) )
				+ ( rectN3[1] * (upEps[1]-rectCenter[1]) )
				+ ( rectN3[2] * (upEps[2]-rectCenter[2]) );

        const double solidAngleUp = solidAngle.SolidAngleRectangleAsArray( rectData, upEps );

        const double hDown = ( rectN3[0] * (downEps[0]-rectCenter[0]) )
				+ ( rectN3[1] * (downEps[1]-rectCenter[1]) )
				+ ( rectN3[2] * (downEps[2]-rectCenter[2]) );

        const double solidAngleDown = solidAngle.SolidAngleRectangleAsArray( rectData, downEps );

    	// compute IqL

        std::pair<KThreeVector, double> IqLFieldAndPotentialUp
			= IqLFieldAndPotential( rectData, upEps, 9, 9, 9 ); /* no line correction */

        std::pair<KThreeVector, double> IqLFieldAndPotentialDown
			= IqLFieldAndPotential( rectData, downEps, 9, 9, 9 ); /* no line correction */

    	const KThreeVector finalField(
    			KEMConstants::OneOverFourPiEps0 * 0.5 * ((rectN3[0]*solidAngleUp + IqLFieldAndPotentialUp.first[0]) + (rectN3[0]*solidAngleDown + IqLFieldAndPotentialDown.first[0])),
				KEMConstants::OneOverFourPiEps0 * 0.5 * ((rectN3[1]*solidAngleUp + IqLFieldAndPotentialUp.first[1]) + (rectN3[1]*solidAngleDown + IqLFieldAndPotentialDown.first[1])),
				KEMConstants::OneOverFourPiEps0 * 0.5 * ((rectN3[2]*solidAngleUp + IqLFieldAndPotentialUp.first[2]) + (rectN3[2]*solidAngleDown + IqLFieldAndPotentialDown.first[2])) );

    	double finalPotential = KEMConstants::OneOverFourPiEps0*0.5 * (
    			(-hUp*solidAngleUp - IqLFieldAndPotentialUp.second)
				+ (-hDown*solidAngleDown - IqLFieldAndPotentialDown.second) );

    	return std::make_pair( finalField, finalPotential );
    }

    const double h = ( rectN3[0] * (P[0]-rectCenter[0]) )
			+ ( rectN3[1] * (P[1]-rectCenter[1]) )
			+ ( rectN3[2] * (P[2]-rectCenter[2]) );

    double fieldPoint[3] = { P[0], P[1], P[2] };

    const double rectSolidAngle = solidAngle.SolidAngleRectangleAsArray( rectData, fieldPoint );

    std::pair<KThreeVector, double> IqLFieldAndPhi
		= IqLFieldAndPotential( rectData, fieldPoint, correctionCounter, correctionLineIndex, distToLineMin );

	const double finalPhi = KEMConstants::OneOverFourPiEps0*((-h*rectSolidAngle) - IqLFieldAndPhi.second);

	const KThreeVector finalField(
			KEMConstants::OneOverFourPiEps0*(rectN3[0]*rectSolidAngle + IqLFieldAndPhi.first[0]),
			KEMConstants::OneOverFourPiEps0*(rectN3[1]*rectSolidAngle + IqLFieldAndPhi.first[1]),
			KEMConstants::OneOverFourPiEps0*(rectN3[2]*rectSolidAngle + IqLFieldAndPhi.first[2]) );

    return std::make_pair( finalField, finalPhi );
}

double KElectrostaticRWGRectangleIntegrator::Potential(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const
{
    double potential = 0.;

    for (KSymmetryGroup<KRectangle>::ShapeCIt it=source->begin();it!=source->end();++it)
        potential += Potential(*it,P);

    return potential;
}

KThreeVector KElectrostaticRWGRectangleIntegrator::ElectricField(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const
{
    KThreeVector electricField(0.,0.,0.);

    for (KSymmetryGroup<KRectangle>::ShapeCIt it=source->begin();it!=source->end();++it)
        electricField += ElectricField(*it,P);

    return electricField;
}

std::pair<KThreeVector, double> KElectrostaticRWGRectangleIntegrator::ElectricFieldAndPotential(const KSymmetryGroup<KRectangle>* source, const KPosition& P) const
{
	std::pair<KThreeVector, double> fieldAndPotential;
    double potential( 0. );
    KThreeVector electricField( 0., 0., 0. );

    for( KSymmetryGroup<KRectangle>::ShapeCIt it=source->begin(); it!=source->end(); ++it ) {
    	fieldAndPotential = ElectricFieldAndPotential( *it, P );
        electricField += fieldAndPotential.first;
    	potential += fieldAndPotential.second;
    }

    return std::make_pair( electricField, potential );
}

}
