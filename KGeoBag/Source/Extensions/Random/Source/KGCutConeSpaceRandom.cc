/*
 * KGCutConeSpaceRandom.cc
 *
 *  Created on: 20.05.2014
 *      Author: user
 */

#include "KGCutConeSpaceRandom.hh"

void KGeoBag::KGCutConeSpaceRandom::VisitCutConeSpace(KGeoBag::KGCutConeSpace* aCutConeSpace) {
	KThreeVector point;

	double gamma = (aCutConeSpace->R2() - aCutConeSpace->R1()) / (aCutConeSpace->Z2() - aCutConeSpace->Z1());
	double lambda = pow(aCutConeSpace->R1() + gamma * (aCutConeSpace->Z2() - aCutConeSpace->Z1()), 3.0)
			- aCutConeSpace->R1() * aCutConeSpace->R1() * aCutConeSpace->R1();

	double z = 1.0 / gamma
			* (pow(aCutConeSpace->R1() * aCutConeSpace->R1() * aCutConeSpace->R1()
					+ lambda * Uniform(), 1.0 / 3.0) - aCutConeSpace->R1())
					+ aCutConeSpace->Z1();

	double r = (aCutConeSpace->R1() + (z - aCutConeSpace->Z1()) * gamma) * sqrt(Uniform());
	double phi = Uniform(0, 2 * KConst::Pi());

	point.SetZ(z);
	point.SetX(cos(phi) * r);
	point.SetY(sin(phi) * r);

	SetRandomPoint(point);
}

