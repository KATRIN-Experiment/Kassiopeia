/*
 * KGDiskSurfaceRandom.cc
 *
 *  Created on: 26.09.2014
 *      Author: J. Behrens
 */

#include "KGDiskSurfaceRandom.hh"

void KGeoBag::KGDiskSurfaceRandom::VisitDiskSurface(KGeoBag::KGDiskSurface* aDiskSpace)
{
    katrin::KThreeVector point;

    // point is always on the disk surface

    double phi = Uniform(0, 2 * katrin::KConst::Pi());
    double r = aDiskSpace->R() * sqrt(Uniform());

    point.SetZ(aDiskSpace->Z());
    point.SetX(cos(phi) * r);
    point.SetY(sin(phi) * r);

    SetRandomPoint(point);
}
