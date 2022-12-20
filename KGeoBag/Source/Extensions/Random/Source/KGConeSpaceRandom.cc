/*
 * KGConeSpaceRandom.cc
 *
 *  Created on: 20.05.2014
 *      Author: user
 */

#include "KGConeSpaceRandom.hh"

void KGeoBag::KGConeSpaceRandom::VisitConeSpace(KGeoBag::KGConeSpace* aConeSpace)
{
    katrin::KThreeVector point;

    // The same like in cut cone with the following conventions:
    // R2 -> RB
    // R1 -> 0
    // Z1 -> ZA
    // Z2 -> ZB
    double gamma = aConeSpace->RB() / (aConeSpace->ZB() - aConeSpace->ZA());
    double lambda = pow(0 + gamma * (aConeSpace->ZB() - aConeSpace->ZA()), 3.0) - 0;

    double z = 1.0 / gamma * (pow(0 + lambda * Uniform(), 1.0 / 3.0) - 0) + aConeSpace->ZA();

    double r = (0 + (z - aConeSpace->ZA()) * gamma) * sqrt(Uniform());
    double phi = Uniform(0, 2 * katrin::KConst::Pi());

    point.SetZ(z);
    point.SetX(cos(phi) * r);
    point.SetY(sin(phi) * r);

    SetRandomPoint(point);
}
