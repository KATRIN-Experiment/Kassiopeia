/*
 * KInducedAzimuthalElectricField.cc
 *
 *  Created on: 15 Apr 2016
 *      Author: wolfgang
 */

#include "KInducedAzimuthalElectricField.hh"
#include "KRampedMagneticField.hh"
#include "KEMSimpleException.hh"

namespace KEMField {

KInducedAzimuthalElectricField::KInducedAzimuthalElectricField() :
        fMagneticField( NULL )
{
}

void KInducedAzimuthalElectricField::SetMagneticField(
        KRampedMagneticField* field) {
    fMagneticField = field;
}

KRampedMagneticField* KInducedAzimuthalElectricField::GetRampedMagneticField() const {
    return fMagneticField;
}

double KInducedAzimuthalElectricField::PotentialCore(
        const KPosition& P, const double& time) const {
    KEMThreeVector electricField = ElectricFieldCore( P, time );
    return -1. * electricField.Dot( P );
}

KEMThreeVector KInducedAzimuthalElectricField::ElectricFieldCore(
        const KPosition& P, const double& time) const {
    double tRadius = P.Perp();
    if (tRadius > 0.)
    {
        KEMThreeVector tAziDirection = 1. / tRadius * KEMThreeVector( -P.Y(), P.X(), 0. );

        KEMThreeVector tMagneticField = fMagneticField->MagneticField( P, time);
        double Modulation = fMagneticField->GetDerivModulationFactor( time );
        return  tAziDirection * (tMagneticField.Z() * (-tRadius / 2.)) * (Modulation * fMagneticField->GetTimeScalingFactor());
    }
    else
    {
        return KEMThreeVector(0.,0.,0.);
    }

}

void KInducedAzimuthalElectricField::InitializeCore()
{
    CheckMagneticField();
    fMagneticField->Initialize();
}

void KInducedAzimuthalElectricField::CheckMagneticField() const {
    if(!fMagneticField)
        throw KEMSimpleException("KInducedAzimuthalElectricField has no magnetic field set as source.");
}

} /* namespace KEMField */
