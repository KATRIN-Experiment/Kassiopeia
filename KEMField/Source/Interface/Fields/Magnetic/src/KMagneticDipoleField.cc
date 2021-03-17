/*
 * KMagneticDipoleField.cc
 *
 *  Created on: 24 Mar 2016
 *      Author: wolfgang
 */

#include "KMagneticDipoleField.hh"

#include "KConst.h"

namespace KEMField
{

KMagneticDipoleField::KMagneticDipoleField() : fLocation(0., 0., 0.), fMoment(0., 0., 0.) {}

KMagneticDipoleField::~KMagneticDipoleField() = default;

/**
 *  The magnetic potential of a dipole at the origin can be written as
 *  A(r) = (m x r)/(|r|^3).
 */
KFieldVector KMagneticDipoleField::MagneticPotentialCore(const KPosition& aSamplePoint) const
{
    KPosition r = aSamplePoint - fLocation;
    double distance = r.Magnitude();
    return fMoment.Cross(r) / (distance * distance * distance);
}

KFieldVector KMagneticDipoleField::MagneticFieldCore(const KPosition& aSamplePoint) const
{
    KFieldVector aPoint = aSamplePoint - fLocation;
    double aPointMag = aPoint.Magnitude();
    double aPointMag2 = aPointMag * aPointMag;
    double aPointMag3 = aPointMag * aPointMag2;
    double aPointMag4 = aPointMag * aPointMag3;
    double aPointMag5 = aPointMag * aPointMag4;

    return (katrin::KConst::MuNull() / (4 * katrin::KConst::Pi())) *
           ((3. / aPointMag5) * fMoment.Dot(aPoint) * aPoint - (1. / aPointMag3) * fMoment);
}

KGradient KMagneticDipoleField::MagneticGradientCore(const KPosition& aSamplePoint) const
{
    KFieldVector aPoint = aSamplePoint - fLocation;
    double aPointMag = aPoint.Magnitude();
    double aPointMag2 = aPointMag * aPointMag;
    double aPointMag3 = aPointMag * aPointMag2;
    double aPointMag4 = aPointMag * aPointMag3;
    double aPointMag5 = aPointMag * aPointMag4;
    double aPointMag6 = aPointMag * aPointMag5;
    double aPointMag7 = aPointMag * aPointMag6;
    double tX = aPoint.X();
    double tY = aPoint.Y();
    double tZ = aPoint.Z();
    double tMX = fMoment.X();
    double tMY = fMoment.Y();
    double tMZ = fMoment.Z();
    double tMR = fMoment.Dot(aPoint);

    KGradient aGradient;

    aGradient(0, 0) = ((3. * katrin::KConst::MuNull()) / (4 * katrin::KConst::Pi())) *
                      ((aPointMag2 - 5. * tX * tX) * (tMR / aPointMag7) + (tMX * tX) * (2. / aPointMag5));
    aGradient(1, 1) = ((3. * katrin::KConst::MuNull()) / (4 * katrin::KConst::Pi())) *
                      ((aPointMag2 - 5. * tY * tY) * (tMR / aPointMag7) + (tMY * tY) * (2. / aPointMag5));
    aGradient(2, 2) = ((3. * katrin::KConst::MuNull()) / (4 * katrin::KConst::Pi())) *
                      ((aPointMag2 - 5. * tZ * tZ) * (tMR / aPointMag7) + (tMZ * tZ) * (2. / aPointMag5));

    aGradient(0, 1) =
        ((3. * katrin::KConst::MuNull()) / (4 * katrin::KConst::Pi())) *
        (tX * tMY * (aPointMag2 - 5. * tY * tY) + tY * tMX * (aPointMag2 - 5. * tX * tX) - 5. * tX * tY * tZ * tMZ) /
        (aPointMag7);
    aGradient(0, 2) =
        ((3. * katrin::KConst::MuNull()) / (4 * katrin::KConst::Pi())) *
        (tX * tMZ * (aPointMag2 - 5. * tZ * tZ) + tZ * tMX * (aPointMag2 - 5. * tX * tX) - 5. * tX * tY * tZ * tMY) /
        (aPointMag7);
    aGradient(1, 2) =
        ((3. * katrin::KConst::MuNull()) / (4 * katrin::KConst::Pi())) *
        (tY * tMZ * (aPointMag2 - 5. * tZ * tZ) + tZ * tMY * (aPointMag2 - 5. * tY * tY) - 5. * tX * tY * tZ * tMX) /
        (aPointMag7);

    aGradient(1, 0) = aGradient(0, 1);
    aGradient(2, 0) = aGradient(0, 2);
    aGradient(2, 1) = aGradient(1, 2);

    return aGradient;
}

void KMagneticDipoleField::SetLocation(const KPosition& aLocation)
{
    fLocation = aLocation;
    return;
}

void KMagneticDipoleField::SetMoment(const KDirection& aMoment)
{
    fMoment = aMoment;
    return;
}

} /* namespace KEMField */
