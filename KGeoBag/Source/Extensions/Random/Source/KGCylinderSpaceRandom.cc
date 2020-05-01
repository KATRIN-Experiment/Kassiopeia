/*
 * KGCylinderSpaceRandom.cc
 *
 *  Created on: 14.05.2014
 *      Author: oertlin
 */

#include "KGCylinderSpaceRandom.hh"

void KGeoBag::KGCylinderSpaceRandom::VisitCylinderSpace(KGeoBag::KGCylinderSpace* aCylinderSpace)
{
    KThreeVector point;

    double z = Uniform();
    double phi = Uniform(0, 2 * katrin::KConst::Pi());
    double r = aCylinderSpace->R() * sqrt(Uniform());

    point.SetZ(aCylinderSpace->Z1() + z * (aCylinderSpace->Z2() - aCylinderSpace->Z1()));
    point.SetX(cos(phi) * r);
    point.SetY(sin(phi) * r);

    SetRandomPoint(point);
}
