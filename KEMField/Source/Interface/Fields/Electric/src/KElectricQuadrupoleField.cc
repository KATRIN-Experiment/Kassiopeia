/*
 * KElectricQuadrupoleField.cc
 *
 *  Created on: 30 Jul 2015
 *      Author: wolfgang
 */

#include "KElectricQuadrupoleField.hh"

#include <limits>

namespace KEMField
{

KElectricQuadrupoleField::KElectricQuadrupoleField() :
    fLocation(katrin::KThreeVector::sZero),
    fStrength(std::numeric_limits<double>::quiet_NaN()),
    fLength(0.),
    fRadius(0.),
    fCharacteristic(0.)
{}

KElectricQuadrupoleField::~KElectricQuadrupoleField() = default;

double KElectricQuadrupoleField::PotentialCore(const KPosition& aSamplePoint) const
{
    // thread-safe
    KPosition FieldPoint = aSamplePoint - fLocation;
    return (fStrength / (2. * fCharacteristic * fCharacteristic)) *
           (FieldPoint[2] * FieldPoint[2] - (1. / 2.) * FieldPoint[0] * FieldPoint[0] -
            (1. / 2.) * FieldPoint[1] * FieldPoint[1]);
}
KFieldVector KElectricQuadrupoleField::ElectricFieldCore(const KPosition& aSamplePoint) const
{
    // thread-safe
    KPosition FieldPoint = aSamplePoint - fLocation;
    KPosition AxialPart = FieldPoint[2] * KPosition(0., 0., 1.);
    KPosition RadialPart = FieldPoint - AxialPart;
    return (fStrength / (2. * fCharacteristic * fCharacteristic)) * RadialPart -
           (fStrength / (fCharacteristic * fCharacteristic)) * AxialPart;
}

void KElectricQuadrupoleField::SetLocation(const KPosition& aLocation)
{
    fLocation = aLocation;
    return;
}
void KElectricQuadrupoleField::SetStrength(const double& aStrength)
{
    fStrength = aStrength;
    return;
}
void KElectricQuadrupoleField::SetLength(const double& aLength)
{
    fLength = aLength;
    fCharacteristic = sqrt((1. / 2.) * (fLength * fLength + (1. / 2.) * fRadius * fRadius));
    return;
}
void KElectricQuadrupoleField::SetRadius(const double& aRadius)
{
    fRadius = aRadius;
    fCharacteristic = sqrt((1. / 2.) * (fLength * fLength + (1. / 2.) * fRadius * fRadius));
    return;
}


} /* namespace KEMField */
