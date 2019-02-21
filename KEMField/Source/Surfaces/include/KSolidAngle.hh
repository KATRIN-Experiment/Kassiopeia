#ifndef KSOLIDANGLE_H_
#define KSOLIDANGLE_H_

#include "KEMConstants.hh"

#include <cmath>

#include "KSurface.hh"
#include "KThreeVector_KEMField.hh"

#define POW2(x) ((x)*(x))

namespace KEMField
{

/**
 * @class KSolidAngle
 *
 * @brief A class for computing solid angles from Euler-Eriksson's formula as described in paper PIER 63, 243-278, 2006
 *
 * @author Daniel Hilk
 */

class KSolidAngle
{
public:
	KSolidAngle();
	virtual ~KSolidAngle();

	double SolidAngleTriangle( const KTriangle* source, KPosition P ) const;
	double SolidAngleTriangleAsArray( const double* data, const double* P ) const;
	double SolidAngleRectangle( const KRectangle* source, KPosition P ) const;
	double SolidAngleRectangleAsArray( const double* data, const double* P ) const;

private:
    const double fMinDistance = 1.e-12; /* for check if field point is on surface */
};


inline double KSolidAngle::SolidAngleTriangle( const KTriangle* source, KPosition P ) const
{
	const double data[11] = {source->GetA(),
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

	const double fieldPoint[3] = { P[0], P[1], P[2] };

	return SolidAngleTriangleAsArray( data, fieldPoint );
}

inline double KSolidAngle::SolidAngleTriangleAsArray( const double* data, const double* P ) const
{
    double res(0.);

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
    const double triMagN3 = 1./sqrt(POW2(triN3[0]) + POW2(triN3[1]) + POW2(triN3[2]));
    triN3[0] = triN3[0]*triMagN3;
    triN3[1] = triN3[1]*triMagN3;
    triN3[2] = triN3[2]*triMagN3;

    // triangle centroid

    const double triCenter[3] = {
    		data[2] + (data[0]*data[5] + data[1]*data[8])/3.,
			data[3] + (data[0]*data[6] + data[1]*data[9])/3.,
			data[4] + (data[0]*data[7] + data[1]*data[10])/3.};

	// quantity h, magnitude corresponds to distance from field point to triangle plane

	const double h = (triN3[0]*(P[0]-triCenter[0]))
			+ (triN3[1]*(P[1]-triCenter[1]))
			+ (triN3[2]*(P[2]-triCenter[2]));

	const double triMagCenterToP = sqrt( POW2(P[0]-triCenter[0])
			+ POW2(P[1]-triCenter[1])
			+ POW2(P[2]-triCenter[2]) );

	if( triMagCenterToP <= fMinDistance )
		res = 2.*KEMConstants::Pi;
	else {
		// unit vectors of distances of corner points to field point in positive rotation order

		double triDistP0Unit[3];
		const double magP0 = 1./sqrt( POW2(triP0[0]-P[0]) + POW2(triP0[1]-P[1]) + POW2(triP0[2]-P[2]) );
		triDistP0Unit[0] = magP0 * (triP0[0]-P[0]);
		triDistP0Unit[1] = magP0 * (triP0[1]-P[1]);
		triDistP0Unit[2] = magP0 * (triP0[2]-P[2]);

		double triDistP1Unit[3];
		const double magP1 = 1./sqrt( POW2(triP1[0]-P[0]) + POW2(triP1[1]-P[1]) + POW2(triP1[2]-P[2]) );
		triDistP1Unit[0] = magP1 * (triP1[0]-P[0]);
		triDistP1Unit[1] = magP1 * (triP1[1]-P[1]);
		triDistP1Unit[2] = magP1 * (triP1[2]-P[2]);

		double triDistP2Unit[3];
		const double magP2 = 1./sqrt( POW2(triP2[0]-P[0]) + POW2(triP2[1]-P[1]) + POW2(triP2[2]-P[2]) );
		triDistP2Unit[0] = magP2 * (triP2[0]-P[0]);
		triDistP2Unit[1] = magP2 * (triP2[1]-P[1]);
		triDistP2Unit[2] = magP2 * (triP2[2]-P[2]);

		const double x = 1.
				+ ( (triDistP0Unit[0]*triDistP1Unit[0])+(triDistP0Unit[1]*triDistP1Unit[1])+(triDistP0Unit[2]*triDistP1Unit[2]) )
				+ ( (triDistP0Unit[0]*triDistP2Unit[0])+(triDistP0Unit[1]*triDistP2Unit[1])+(triDistP0Unit[2]*triDistP2Unit[2]) )
				+ ( (triDistP1Unit[0]*triDistP2Unit[0])+(triDistP1Unit[1]*triDistP2Unit[1])+(triDistP1Unit[2]*triDistP2Unit[2]) );

		// cross product
		const double a12[3] = {
				(triDistP1Unit[1]*triDistP2Unit[2]) - (triDistP1Unit[2]*triDistP2Unit[1]),
				(triDistP1Unit[2]*triDistP2Unit[0]) - (triDistP1Unit[0]*triDistP2Unit[2]),
				(triDistP1Unit[0]*triDistP2Unit[1]) - (triDistP1Unit[1]*triDistP2Unit[0])
		};

		const double y = fabs( ( (triDistP0Unit[0]*a12[0]) + (triDistP0Unit[1]*a12[1]) + (triDistP0Unit[2]*a12[2]) ) );

		res = fabs( 2.*atan2(y,x) );
	}

	if( h < 0. ) res *= -1.;

	return res;
}

inline double KSolidAngle::SolidAngleRectangle( const KRectangle* source, KPosition P ) const
{
	const double data[11] = {source->GetA(),
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

	const double fieldPoint[3] = { P[0], P[1], P[2] };

	return SolidAngleRectangleAsArray( data, fieldPoint );
}

inline double KSolidAngle::SolidAngleRectangleAsArray( const double* data, const double* P ) const
{
	// Computing solid angle from two triangles:
    // Triangle 1: P0 - P1 (N1) - P2 (N2)
    // Triangle 2: P2 - P3 (-N1) - P0 (-N2)

    double res(0.);

    // corner points P0, P1, P2 and P3

    const double rectP0[3] = { data[2], data[3], data[4] };

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

    const double rectCenter[3] = {
    		data[2] + data[0]*data[5]*.5 + data[1]*data[8]*.5,
			data[3] + data[0]*data[6]*.5 + data[1]*data[9]*.5,
			data[4] + data[0]*data[7]*.5 + data[1]*data[10]*.5 };

    const double rectN3[3] = {
    		data[6]*data[10] - data[7]*data[9],
			data[7]*data[8]  - data[5]*data[10],
			data[5]*data[9]  - data[6]*data[8] };

	// quantity h, magnitude corresponds to distance from field point to triangle plane

	const double h = (rectN3[0]*(P[0]-rectCenter[0])) + (rectN3[1]*(P[1]-rectCenter[1])) + (rectN3[2]*(P[2]-rectCenter[2]));

	// check if field point is on rectangle plane

	if( fabs(h) < fMinDistance ) {

		// check if field point is inside the rectangle plane

  		// line 1: P0 - P1, line 2: P0 - P3

		const double rectSide1[3] = {
  				data[0]*data[5],
				data[0]*data[6],
				data[0]*data[7]
  		};
  		const double rectSide2[3] = {
  				data[1]*data[8],
				data[1]*data[9],
				data[1]*data[10]
  		};

  		const double rectDistP[3] = {
  				data[2] - P[0],
				data[3] - P[1],
				data[4] - P[2]
  		};

    	// parameter lambda is needed for checking if point is on rectangle surface
    	// here: factor -1 for turning direction of reDistP

    	const double lineLambda1 = (-1) * ((rectDistP[0]*rectSide1[0])+(rectDistP[1]*rectSide1[1])+(rectDistP[2]*rectSide1[2])) / POW2(data[0]);
    	const double lineLambda2 = (-1) * ((rectDistP[0]*rectSide2[0])+(rectDistP[1]*rectSide2[1])+(rectDistP[2]*rectSide2[2])) / POW2(data[1]);

    	// field point is on rectangle plane

		if( lineLambda1>0. && lineLambda1<1. && lineLambda2>0. && lineLambda2<1. )
			res = 2.*KEMConstants::Pi;
		else // field point is in rectangle plane, but outside the surface
			res = 0.;
	}
	else {
		// unit vectors of distances of corner points to field point in positive rotation order

		double rectDistP0Unit[3];
		const double magP0 = 1./sqrt( POW2(rectP0[0]-P[0])
				+ POW2(rectP0[1]-P[1])
				+ POW2(rectP0[2]-P[2]) );
		rectDistP0Unit[0] = magP0 * (rectP0[0]-P[0]);
		rectDistP0Unit[1] = magP0 * (rectP0[1]-P[1]);
		rectDistP0Unit[2] = magP0 * (rectP0[2]-P[2]);

		double rectDistP1Unit[3];
		const double magP1 = 1./sqrt( POW2(rectP1[0]-P[0])
				+ POW2(rectP1[1]-P[1])
				+ POW2(rectP1[2]-P[2]) );
		rectDistP1Unit[0] = magP1 * (rectP1[0]-P[0]);
		rectDistP1Unit[1] = magP1 * (rectP1[1]-P[1]);
		rectDistP1Unit[2] = magP1 * (rectP1[2]-P[2]);

		double rectDistP2Unit[3];
		const double magP2 = 1./sqrt( POW2(rectP2[0]-P[0])
				+ POW2(rectP2[1]-P[1])
				+ POW2(rectP2[2]-P[2]) );
		rectDistP2Unit[0] = magP2 * (rectP2[0]-P[0]);
		rectDistP2Unit[1] = magP2 * (rectP2[1]-P[1]);
		rectDistP2Unit[2] = magP2 * (rectP2[2]-P[2]);

		double rectDistP3Unit[3];
		const double magP3 = 1./sqrt( POW2(rectP3[0]-P[0])
				+ POW2(rectP3[1]-P[1])
				+ POW2(rectP3[2]-P[2]) );
		rectDistP3Unit[0] = magP3 * (rectP3[0]-P[0]);
		rectDistP3Unit[1] = magP3 * (rectP3[1]-P[1]);
		rectDistP3Unit[2] = magP3 * (rectP3[2]-P[2]);

		const double x1 = 1.
				+ ( (rectDistP0Unit[0]*rectDistP1Unit[0])+(rectDistP0Unit[1]*rectDistP1Unit[1])+(rectDistP0Unit[2]*rectDistP1Unit[2]) )
				+ ( (rectDistP0Unit[0]*rectDistP2Unit[0])+(rectDistP0Unit[1]*rectDistP2Unit[1])+(rectDistP0Unit[2]*rectDistP2Unit[2]) )
				+ ( (rectDistP1Unit[0]*rectDistP2Unit[0])+(rectDistP1Unit[1]*rectDistP2Unit[1])+(rectDistP1Unit[2]*rectDistP2Unit[2]) );

		// cross product

		const double a12[3] = {
				(rectDistP1Unit[1]*rectDistP2Unit[2]) - (rectDistP1Unit[2]*rectDistP2Unit[1]),
				(rectDistP1Unit[2]*rectDistP2Unit[0]) - (rectDistP1Unit[0]*rectDistP2Unit[2]),
				(rectDistP1Unit[0]*rectDistP2Unit[1]) - (rectDistP1Unit[1]*rectDistP2Unit[0])
		};

		const double y1 = fabs( (rectDistP0Unit[0]*a12[0])+(rectDistP0Unit[1]*a12[1])+(rectDistP0Unit[2]*a12[2]) );

		const double solidAngle1 = fabs( 2.*atan2(y1,x1) );

		const double x2 = 1.
				+ ( (rectDistP2Unit[0]*rectDistP3Unit[0])+(rectDistP2Unit[1]*rectDistP3Unit[1])+(rectDistP2Unit[2]*rectDistP3Unit[2]) )
				+ ( (rectDistP2Unit[0]*rectDistP0Unit[0])+(rectDistP2Unit[1]*rectDistP0Unit[1])+(rectDistP2Unit[2]*rectDistP0Unit[2]) )
				+ ( (rectDistP3Unit[0]*rectDistP0Unit[0])+(rectDistP3Unit[1]*rectDistP0Unit[1])+(rectDistP3Unit[2]*rectDistP0Unit[2]) );

		// cross product

		const double a30[3] = {
				(rectDistP3Unit[1]*rectDistP0Unit[2]) - (rectDistP3Unit[2]*rectDistP0Unit[1]),
				(rectDistP3Unit[2]*rectDistP0Unit[0]) - (rectDistP3Unit[0]*rectDistP0Unit[2]),
				(rectDistP3Unit[0]*rectDistP0Unit[1]) - (rectDistP3Unit[1]*rectDistP0Unit[0])
		};

		const double y2 = fabs( (rectDistP2Unit[0]*a30[0])+(rectDistP2Unit[1]*a30[1])+(rectDistP2Unit[2]*a30[2]) );

		const double solidAngle2 = fabs( 2.*atan2(y2,x2) );

		res = solidAngle1 + solidAngle2;
	}

	if( h < 0. ) res *= -1.;

	return res;
}

}

#endif
