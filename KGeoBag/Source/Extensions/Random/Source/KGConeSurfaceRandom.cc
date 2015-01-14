/*
 * KGConeSurfaceRandom.cc
 *
 *  Created on: 20.05.2014
 *      Author: user
 */

#include "KGConeSurfaceRandom.hh"

void KGeoBag::KGConeSurfaceRandom::VisitConeSurface(KGeoBag::KGConeSurface* aConeSpace) {
	KThreeVector point;

	// Decide, on which area the point have to be
	double h = abs(aConeSpace->ZA() - aConeSpace->ZB());
	double face = KConst::Pi() * aConeSpace->RB() * aConeSpace->RB();
	double curvedSurfaceArea = KConst::Pi() * aConeSpace->RB() * (aConeSpace->RB()
			+ sqrt(aConeSpace->RB() * aConeSpace->RB() + h * h));

	double total = face + curvedSurfaceArea;
	double decision = Uniform(0, total);
	double phi = Uniform(0, 2 * KConst::Pi());

	if((total -= curvedSurfaceArea) < decision) {
		double z1 = aConeSpace->ZA();
		double z2 = aConeSpace->ZB();
		double z = sqrt(Uniform() * (z2 * z2 - z1 * z1) + z1 * z1);
		double r = aConeSpace->RB() / (aConeSpace->ZB() - aConeSpace->ZA()) * z;

		point.SetZ(z);
		point.SetX(cos(phi) * r);
		point.SetY(sin(phi) * r);
	} else {
		double r = aConeSpace->RB() * sqrt(Uniform());

		point.SetZ(aConeSpace->ZB());
		point.SetX(cos(phi) * r);
		point.SetY(sin(phi) * r);
	}

	SetRandomPoint(point);
}
