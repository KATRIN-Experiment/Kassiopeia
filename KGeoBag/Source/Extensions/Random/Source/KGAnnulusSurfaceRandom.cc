/*
 * KGAnnulusSurfaceRandom.cc
 *
 *  Created on: 08.10.2020
 *      Author: J. Behrens
 */

#include "KGAnnulusSurfaceRandom.hh"

void KGeoBag::KGAnnulusSurfaceRandom::VisitAnnulusSurface(KGeoBag::KGAnnulusSurface* anAnnulusSpace)
{
    katrin::KThreeVector point;

    // point is always on the disk surface

    double phi = Uniform(0, 2 * katrin::KConst::Pi());
    double w = anAnnulusSpace->R2() - anAnnulusSpace->R1();
    double r = w * sqrt(Uniform()) + anAnnulusSpace->R1();

    point.SetZ(anAnnulusSpace->Z());
    point.SetX(cos(phi) * r);
    point.SetY(sin(phi) * r);

    SetRandomPoint(point);
}
