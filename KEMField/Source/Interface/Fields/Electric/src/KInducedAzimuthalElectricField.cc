/*
 * KInducedAzimuthalElectricField.cc
 *
 *  Created on: 15 Apr 2016
 *      Author: wolfgang
 */

#include "KInducedAzimuthalElectricField.hh"

#include "KEMSimpleException.hh"
#include "KRampedMagneticField.hh"

namespace KEMField
{

KInducedAzimuthalElectricField::KInducedAzimuthalElectricField() : fMagneticField(nullptr) {}

void KInducedAzimuthalElectricField::SetMagneticField(KRampedMagneticField* field)
{
    fMagneticField = field;
}

KRampedMagneticField* KInducedAzimuthalElectricField::GetRampedMagneticField() const
{
    return fMagneticField;
}

double KInducedAzimuthalElectricField::PotentialCore(const KPosition& P, const double& time) const
{
    KFieldVector electricField = ElectricFieldCore(P, time);
    return -1. * electricField.Dot(P);
}

KFieldVector KInducedAzimuthalElectricField::ElectricFieldCore(const KPosition& P, const double& time) const
{
    double tRadius = P.Perp();
    if (tRadius > 0.) {
        KFieldVector tAziDirection = 1. / tRadius * KFieldVector(-P.Y(), P.X(), 0.);

        KFieldVector tMagneticField = fMagneticField->MagneticField(P, time);
        double Modulation = fMagneticField->GetDerivModulationFactor(time);
        return tAziDirection * (tMagneticField.Z() * (-tRadius / 2.)) *
               (Modulation * fMagneticField->GetTimeScalingFactor());
    }
    else {
        return KFieldVector(0., 0., 0.);
    }
}

void KInducedAzimuthalElectricField::InitializeCore()
{
    CheckMagneticField();
    fMagneticField->Initialize();
}

void KInducedAzimuthalElectricField::CheckMagneticField() const
{
    if (!fMagneticField)
        throw KEMSimpleException("KInducedAzimuthalElectricField has no magnetic field set as source.");
}

} /* namespace KEMField */
