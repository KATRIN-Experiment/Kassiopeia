/*
 * KGCylinderSurfaceRandom.cc
 *
 *  Created on: 20.05.2014
 *      Author: user
 */

#include "KGCylinderSurfaceRandom.hh"

void KGeoBag::KGCylinderSurfaceRandom::VisitCylinderSurface(KGeoBag::KGCylinderSurface* aCylinderSpace)
{
    KThreeVector point;


    double phi = Uniform(0, 2 * katrin::KConst::Pi());

    double z = Uniform();

    point.SetZ(aCylinderSpace->Z1() + z * (aCylinderSpace->Z2() - aCylinderSpace->Z1()));
    point.SetX(cos(phi) * aCylinderSpace->R());
    point.SetY(sin(phi) * aCylinderSpace->R());


    SetRandomPoint(point);
}
