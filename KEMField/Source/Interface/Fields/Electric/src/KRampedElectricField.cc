/*
 * KRampedElectricField.cc
 *
 *  Created on: 31 May 2016
 *      Author: wolfgang
 */

#include "KRampedElectricField.hh"

#include "KEMCout.hh"
#include "KEMSimpleException.hh"

#include <cassert>


namespace KEMField
{

KRampedElectricField::KRampedElectricField() :
    fRootElectricField(nullptr),
    fRampingType(rtExponential),
    fNumCycles(1),
    fRampUpDelay(0.),
    fRampDownDelay(0.),
    fRampUpTime(0.),
    fRampDownTime(0.),
    fTimeConstant(0.),
    fTimeScalingFactor(1.)
{}

KRampedElectricField::~KRampedElectricField() = default;

double KRampedElectricField::PotentialCore(const KPosition& aSamplePoint, const double& aSampleTime) const
{
    double potential = fRootElectricField->Potential(aSamplePoint, aSampleTime);
    double Modulation = GetModulationFactor(aSampleTime);
    return potential * Modulation;
    //fieldmsg_debug( "Ramped electric field <" << GetName() << "> returns U=" << aTarget << " at t=" << aSampleTime << eom );
}

KFieldVector KRampedElectricField::ElectricFieldCore(const KPosition& aSamplePoint, const double& aSampleTime) const
{
    KFieldVector field = fRootElectricField->ElectricField(aSamplePoint, aSampleTime);
    double Modulation = GetModulationFactor(aSampleTime);
    return field * Modulation;
    //fieldmsg_debug( "Ramped electric field <" << GetName() << "> returns E=" << aTarget << " at t=" << aSampleTime << eom );
}

double KRampedElectricField::GetModulationFactor(const double& aTime) const
{
    double tLength = fRampUpDelay + fRampUpTime + fRampDownDelay + fRampDownTime;

    double tTime = aTime;
    auto tCycle = (int) floor(tTime / tLength);
    if (tCycle < fNumCycles)
        tTime -= tCycle * tLength;
    tTime *= fTimeScalingFactor;

    double tUp = fRampUpDelay * fTimeScalingFactor;
    double tHigh = tUp + fRampUpTime * fTimeScalingFactor;
    double tDown = tHigh + fRampDownDelay * fTimeScalingFactor;
    double tLow = tDown + fRampDownTime * fTimeScalingFactor;
    double tOmega = 2. * M_PI / tLength;

    double Field = 0.;
    switch (fRampingType) {
        case rtLinear:
            if (tTime >= tLow)
                Field = 0.;
            else if (tTime >= tDown)
                Field = (1. - ((tTime - tDown) / fRampDownTime));
            else if (tTime >= tHigh)
                Field = 1.;
            else if (tTime >= tUp)
                Field = ((tTime - tUp) / fRampUpTime);
            break;

        case rtExponential:
            if (tTime >= tLow)
                Field = 0.;
            else if (tTime >= tDown) {
                Field = exp(-(tTime - tDown) / fTimeConstant);
            }
            else if (tTime >= tHigh)
                Field = 1.;
            else if (tTime >= tUp)
                Field = 1. - exp(-(tTime - tUp) / fTimeConstant);
            break;

        case rtSinus:
            if (tTime >= tUp)
                Field = sin(tTime * tOmega);
            break;

        default:
            throw KEMSimpleException("Ramped Electric Field: Specified ramping type is not implemented");
            break;
    }

    //fieldmsg_debug( "Ramped electric field <" << GetName() << "> uses modulation factor " << Field << " at t=" << tTime << eom );
    return Field;
}

void KRampedElectricField::InitializeCore()
{
    //fieldmsg_debug( "Initializing root electric field <" << fRootElectricFieldName << "> from ramped electric field <" << GetName() << ">" << eom

    assert(fNumCycles > 0);
    assert(fTimeScalingFactor > 0.);
    assert(fRampUpTime >= 0.);
    //fieldmsg_assert( fRampUpDelay, >= 0. ); ramping can start before t=0
    assert(fRampDownTime >= 0.);
    //fieldmsg_assert( fRampDownDelay, >= 0. ); ramping back can also start before t=0
    if (fRampingType == rtExponential) {
        assert(fTimeConstant > 0.);
    }

    fRootElectricField->Initialize();
}

} /* namespace KEMField */
