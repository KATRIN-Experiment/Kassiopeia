/*
 * KRampedMagneticField.cc
 *
 *  Created on: 4 Apr 2016
 *      Author: wolfgang
 */

#include "KRampedMagneticField.hh"

#include "KEMCout.hh"

#include <cassert>

namespace KEMField
{


KRampedMagneticField::KRampedMagneticField() :
    fRootMagneticField(nullptr),
    fRampingType(rtExponential),
    fNumCycles(1),
    fRampUpDelay(0.),
    fRampDownDelay(0.),
    fRampUpTime(0.),
    fRampDownTime(0.),
    fTimeConstant(1.),
    fTimeConstant2(1.),
    fTimeScalingFactor(1.)
{}

KRampedMagneticField::~KRampedMagneticField() {}

KThreeVector KRampedMagneticField::MagneticPotentialCore(const KPosition& aSamplePoint, const double& aSampleTime) const
{
    KThreeVector aPotential = fRootMagneticField->MagneticPotential(aSamplePoint, aSampleTime);
    double Modulation = GetModulationFactor(aSampleTime);
    return aPotential * Modulation;
}

KThreeVector KRampedMagneticField::MagneticFieldCore(const KPosition& aSamplePoint, const double& aSampleTime) const
{
    KThreeVector aField = fRootMagneticField->MagneticField(aSamplePoint, aSampleTime);
    double Modulation = GetModulationFactor(aSampleTime);
    return aField * Modulation;
}

KGradient KRampedMagneticField::MagneticGradientCore(const KPosition& aSamplePoint, const double& aSampleTime) const
{
    KGradient aGradient = fRootMagneticField->MagneticGradient(aSamplePoint, aSampleTime);
    double Modulation = GetModulationFactor(aSampleTime);
    return aGradient * Modulation;
}

double KRampedMagneticField::GetModulationFactor(const double& aTime) const
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

        case rtInversion:
            if (tTime >= tLow)
                Field = 0.;
            else if (tTime >= tDown) {
                Field = 1. - 2. * exp(-(tTime - tDown) / fTimeConstant);
            }
            else if (tTime >= tHigh)
                Field = 1.;
            else if (tTime >= tUp)
                Field = -1. + 2. * exp(-(tTime - tUp) / fTimeConstant);
            break;

        case rtInversion2:
            if (tTime >= tLow)
                Field = 1.;
            else if (tTime >= tDown)  // back to normal
                Field = 1. - (exp(-(tTime - tDown) / fTimeConstant) + exp(-(tTime - tDown) / fTimeConstant2));
            else if (tTime >= tHigh)
                Field = -1.;
            else if (tTime >= tUp)  // to inverted
                Field = -1. + (exp(-(tTime - tUp) / fTimeConstant) + exp(-(tTime - tUp) / fTimeConstant2));
            break;

        case rtFlipBox:
            if (tTime >= tLow)  // normal
                Field = 1.;
            else if (tTime >= tDown)  // ramp to normal
            {
                double tMid = 0.5 * (tLow + tDown);
                if (tTime < tMid)
                    Field = -1. * exp(-(tTime - tDown) / fTimeConstant);
                else
                    Field = 1. - exp(-(tTime - tMid) / fTimeConstant);
            }
            else if (tTime >= tHigh)  // inverted
                Field = -1.;
            else if (tTime >= tUp)  // ramp to inverted
            {
                double tMid = 0.5 * (tHigh + tUp);
                if (tTime < tMid)
                    Field = exp(-(tTime - tUp) / fTimeConstant);
                else
                    Field = -1. + exp(-(tTime - tMid) / fTimeConstant);
            }
            break;

        default:
            cout << "KRampedMagneticField: Specified ramping type is not implemented" << endl;
            exit(-1);
            break;
    }
    return Field;
}

double KRampedMagneticField::GetDerivModulationFactor(const double& aTime) const
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

    double DerivField = 0.;
    switch (fRampingType) {
        case rtLinear:
            if (tTime >= tLow)
                DerivField = 0.;
            else if (tTime >= tDown)
                DerivField = (-1. / fRampDownTime);
            else if (tTime >= tHigh)
                DerivField = 0.;
            else if (tTime >= tUp)
                DerivField = (1. / fRampUpTime);
            break;

        case rtExponential:
            if (tTime >= tLow)
                DerivField = 0.;
            else if (tTime >= tDown)
                DerivField = 1. / fTimeConstant * exp(-(tTime - tDown) / fTimeConstant);
            else if (tTime >= tHigh)
                DerivField = 0.;
            else if (tTime >= tUp)
                DerivField = -1. / fTimeConstant * exp(-(tTime - tUp) / fTimeConstant);
            break;

        case rtInversion:
            if (tTime >= tLow)
                DerivField = 0.;
            else if (tTime >= tDown)
                DerivField = 2. / fTimeConstant * exp(-(tTime - tDown) / fTimeConstant);
            else if (tTime >= tHigh)
                DerivField = 0.;
            else if (tTime >= tUp)
                DerivField = -2. / fTimeConstant * exp(-(tTime - tUp) / fTimeConstant);
            break;

        case rtInversion2:
            if (tTime >= tLow)
                DerivField = 0.;
            else if (tTime >= tDown)
                DerivField = 1. / fTimeConstant * exp(-(tTime - tDown) / fTimeConstant) +
                             1. / fTimeConstant2 * exp(-(tTime - tDown) / fTimeConstant2);
            else if (tTime >= tHigh)
                DerivField = 0.;
            else if (tTime >= tUp)
                DerivField = -1. / fTimeConstant * exp(-(tTime - tUp) / fTimeConstant) -
                             1. / fTimeConstant2 * exp(-(tTime - tUp) / fTimeConstant2);
            break;

        case rtFlipBox:
            if (tTime >= tLow)  // normal
                DerivField = 0.;
            else if (tTime >= tDown)  // ramp to normal
            {
                double tMid = 0.5 * (tLow + tDown);
                if (tTime < tMid)
                    DerivField = -1. / fTimeConstant * exp(-(tTime - tDown) / fTimeConstant);
                else
                    DerivField = -1. / fTimeConstant * exp(-(tTime - tMid) / fTimeConstant);
            }
            else if (tTime >= tHigh)  // inverted
                DerivField = 0.;
            else if (tTime >= tUp)  // ramp to inverted
            {
                double tMid = 0.5 * (tHigh + tUp);
                if (tTime < tMid)
                    DerivField = 1. / fTimeConstant * exp(-(tTime - tUp) / fTimeConstant);
                else
                    DerivField = 1. / fTimeConstant * exp(-(tTime - tMid) / fTimeConstant);
            }
            break;

        default:
            cout << "Specified ramping type is not implemented" << endl;
            exit(-1);
            break;
    }
    return DerivField;
}

void KRampedMagneticField::InitializeCore()
{
    assert(fNumCycles > 0);
    assert(fTimeScalingFactor > 0.);
    assert(fRampUpTime >= 0.);
    //    assert( fRampUpDelay >= 0. ); ramping can start before t=0
    assert(fRampDownTime >= 0.);
    //    assert( fRampDownDelay >= 0. ); ramping back can also start before t=0
    if (fRampingType == rtExponential || fRampingType == rtInversion || fRampingType == rtInversion2 ||
        fRampingType == rtFlipBox)
        assert(fTimeConstant > 0.);

    if (fRampingType == rtInversion2) {
        assert(fTimeConstant2 > 0.);
    }

    fRootMagneticField->Initialize();
}

} /* namespace KEMField */
