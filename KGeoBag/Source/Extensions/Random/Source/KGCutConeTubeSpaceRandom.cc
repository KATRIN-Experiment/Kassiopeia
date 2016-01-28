/*
 * KGCutConeTubeSpaceRandom.cc
 *
 *  Created on: 16.09.2015
 *      Author: Daniel Hilk
 */

#include "KGCutConeTubeSpaceRandom.hh"


double KGeoBag::KGCutConeTubeSpaceRandom::LinearInterpolation( double zInput, const double z1, const double r1, const double z2, const double r2 )
{
    double f = (r2-r1) / (z2 - z1) * zInput + ((z2*r1) - (z1*r2)) / (z2-z1);
    return f;
}


void KGeoBag::KGCutConeTubeSpaceRandom::VisitCutConeTubeSpace(KGeoBag::KGCutConeTubeSpace* aCutConeTubeSpace)
{
	KThreeVector point;

	const double dist1 = fabs( aCutConeTubeSpace->R12() - aCutConeTubeSpace->R11() );
	const double dist2 = fabs( aCutConeTubeSpace->R22() - aCutConeTubeSpace->R21() );
	const double dist1pow3 = dist1 * dist1 * dist1;
	double zRnd(0.);

	if( dist2 != dist1 ) {
		const double gamma = (dist2 - dist1) / (aCutConeTubeSpace->Z2() - aCutConeTubeSpace->Z1());
		const double lambda = pow(dist1 + gamma * (aCutConeTubeSpace->Z2() - aCutConeTubeSpace->Z1()), 3.0) - dist1pow3;
		zRnd = 1.0 / gamma * (pow(dist1pow3 + lambda * Uniform(), 1.0 / 3.0) - dist1) + aCutConeTubeSpace->Z1();
	}
	else {
		zRnd = Uniform( aCutConeTubeSpace->Z1(), aCutConeTubeSpace->Z2() );
	}
	point.SetZ( zRnd );

	// minimal r position at z = zRnd
	const double rMin = LinearInterpolation( zRnd, aCutConeTubeSpace->Z1(), aCutConeTubeSpace->R11(), aCutConeTubeSpace->Z2(), aCutConeTubeSpace->R21() );
	// maximal r position at z = zRnd
	const double rMax = LinearInterpolation( zRnd, aCutConeTubeSpace->Z1(), aCutConeTubeSpace->R12(), aCutConeTubeSpace->Z2(), aCutConeTubeSpace->R22() );

	const double rRnd = sqrt( Uniform( rMin*rMin, rMax*rMax ) );

	const double phiRnd = Uniform( 0, 2 * KConst::Pi() );

	point.SetZ( zRnd );
	point.SetX( cos(phiRnd) * rRnd);
	point.SetY( sin(phiRnd) * rRnd);

	SetRandomPoint(point);
}

