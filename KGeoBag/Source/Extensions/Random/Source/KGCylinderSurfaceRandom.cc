/*
 * KGCylinderSurfaceRandom.cc
 *
 *  Created on: 20.05.2014
 *      Author: user
 */

#include "KGCylinderSurfaceRandom.hh"

void KGeoBag::KGCylinderSurfaceRandom::VisitCylinderSurface(KGeoBag::KGCylinderSurface* aCylinderSpace) {
	KThreeVector point;

	// Decide, on which area the point have to be
	double face = KConst::Pi() * aCylinderSpace->R() * aCylinderSpace->R();
	double curvedSurfaceArea = 2 * KConst::Pi() * aCylinderSpace->R()
			* abs(aCylinderSpace->Z1() - aCylinderSpace->Z2());

	double total = 2 * face + curvedSurfaceArea;
	double decision = Uniform(0, total);

	double phi = Uniform(0, 2 * KConst::Pi());

		double z = Uniform();

		point.SetZ(aCylinderSpace->Z1() + z * (aCylinderSpace->Z2() - aCylinderSpace->Z1()));
		point.SetX(cos(phi) * aCylinderSpace->R());
		point.SetY(sin(phi) * aCylinderSpace->R());


	SetRandomPoint(point);
}

