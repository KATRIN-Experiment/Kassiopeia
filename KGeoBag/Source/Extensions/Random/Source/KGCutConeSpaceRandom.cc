/*
 * KGCutConeSpaceRandom.cc
 *
 *  Created on: 20.05.2014
 *      Author: user
 */

#include "KGCutConeSpaceRandom.hh"

void KGeoBag::KGCutConeSpaceRandom::VisitCutConeSpace(KGeoBag::KGCutConeSpace* aCutConeSpace) {
	KThreeVector point;

	double z(0.), r(0.);

	if( aCutConeSpace->R2() != aCutConeSpace->R1() ) {
		double R1pow3 = aCutConeSpace->R1() * aCutConeSpace->R1() * aCutConeSpace->R1();
		double gamma = (aCutConeSpace->R2() - aCutConeSpace->R1()) / (aCutConeSpace->Z2() - aCutConeSpace->Z1());
		double lambda = pow(aCutConeSpace->R1() + gamma * (aCutConeSpace->Z2() - aCutConeSpace->Z1()), 3.0) - R1pow3;

		z = 1.0 / gamma * (pow(R1pow3 + lambda * Uniform(), 1.0 / 3.0) - aCutConeSpace->R1()) + aCutConeSpace->Z1();
		r = (aCutConeSpace->R1() + (z - aCutConeSpace->Z1()) * gamma) * sqrt(Uniform());
	}
	else {
		z = aCutConeSpace->Z1() + z * (aCutConeSpace->Z2() - aCutConeSpace->Z1());
		r = aCutConeSpace->R1() * sqrt(Uniform());
	}

	double phi = Uniform(0, 2 * KConst::Pi());

	point.SetZ(z);
	point.SetX(cos(phi) * r);
	point.SetY(sin(phi) * r);

	SetRandomPoint(point);
}

