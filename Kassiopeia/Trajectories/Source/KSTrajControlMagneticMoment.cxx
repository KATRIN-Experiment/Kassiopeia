#include "KSTrajControlMagneticMoment.h"

#include "KSTrajectoriesMessage.h"

namespace Kassiopeia
{

KSTrajControlMagneticMoment::KSTrajControlMagneticMoment() :
    fLowerLimit(1.e-14),
    fUpperLimit(1.e-10),
    fTimeStep(0.),
    fFirstStep(true)
{}
KSTrajControlMagneticMoment::KSTrajControlMagneticMoment(const KSTrajControlMagneticMoment& aCopy) :
    KSComponent(),
    fLowerLimit(aCopy.fLowerLimit),
    fUpperLimit(aCopy.fUpperLimit),
    fTimeStep(aCopy.fTimeStep),
    fFirstStep(true)
{}
KSTrajControlMagneticMoment* KSTrajControlMagneticMoment::Clone() const
{
    return new KSTrajControlMagneticMoment(*this);
}
KSTrajControlMagneticMoment::~KSTrajControlMagneticMoment() {}

void KSTrajControlMagneticMoment::ActivateObject()
{
    trajmsg_debug("stepsize magnetic moment resetting" << eom);
    fFirstStep = true;
    return;
}

void KSTrajControlMagneticMoment::Calculate(const KSTrajExactParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize magnetic moment on first step" << eom);
        fTimeStep = 0.0625 / aParticle.GetCyclotronFrequency();
        fFirstStep = false;
    }

    trajmsg_debug("stepsize magnetic moment suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}
void KSTrajControlMagneticMoment::Check(const KSTrajExactParticle& anInitialParticle,
                                        const KSTrajExactParticle& aFinalParticle, const KSTrajExactError&, bool& aFlag)
{
    double tFinalMagneticMoment = aFinalParticle.GetOrbitalMagneticMoment();
    double tInitialMagneticMoment = anInitialParticle.GetOrbitalMagneticMoment();
    double tAdiabaticityViolation =
        fabs(2. * (tFinalMagneticMoment - tInitialMagneticMoment) / (tFinalMagneticMoment + tInitialMagneticMoment));

    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

    if (tAdiabaticityViolation < fLowerLimit) {
        trajmsg_debug("stepsize magnetic moment increasing stepsize at violation <" << tAdiabaticityViolation << ">"
                                                                                    << eom);
        fTimeStep = 1.5 * fTimeStep;
        aFlag = true;
        return;
    }

    if (tAdiabaticityViolation > fUpperLimit) {
        trajmsg_debug("stepsize magnetic moment decreasing stepsize at violation <" << tAdiabaticityViolation << ">"
                                                                                    << eom);
        fTimeStep = 0.4 * fTimeStep;
        aFlag = false;
        return;
    }

    trajmsg_debug("stepsize magnetic moment keeping stepsize at violation <" << tAdiabaticityViolation << ">" << eom);
    aFlag = true;
    return;
}

void KSTrajControlMagneticMoment::Calculate(const KSTrajExactSpinParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize magnetic moment on first step" << eom);
        fTimeStep = 0.0625 / aParticle.GetCyclotronFrequency();
        fFirstStep = false;
    }

    trajmsg_debug("stepsize magnetic moment suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}
void KSTrajControlMagneticMoment::Check(const KSTrajExactSpinParticle& anInitialParticle,
                                        const KSTrajExactSpinParticle& aFinalParticle, const KSTrajExactSpinError&,
                                        bool& aFlag)
{
    double tFinalMagneticMoment = aFinalParticle.GetOrbitalMagneticMoment();
    double tInitialMagneticMoment = anInitialParticle.GetOrbitalMagneticMoment();
    double tAdiabaticityViolation =
        fabs(2. * (tFinalMagneticMoment - tInitialMagneticMoment) / (tFinalMagneticMoment + tInitialMagneticMoment));

    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

    if (tAdiabaticityViolation < fLowerLimit) {
        trajmsg_debug("stepsize magnetic moment increasing stepsize at violation <" << tAdiabaticityViolation << ">"
                                                                                    << eom);
        fTimeStep = 1.5 * fTimeStep;
        aFlag = true;
        return;
    }

    if (tAdiabaticityViolation > fUpperLimit) {
        trajmsg_debug("stepsize magnetic moment decreasing stepsize at violation <" << tAdiabaticityViolation << ">"
                                                                                    << eom);
        fTimeStep = 0.4 * fTimeStep;
        aFlag = false;
        return;
    }

    trajmsg_debug("stepsize magnetic moment keeping stepsize at violation <" << tAdiabaticityViolation << ">" << eom);
    aFlag = true;
    return;
}

void KSTrajControlMagneticMoment::Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize magnetic moment on first step" << eom);
        fTimeStep = 0.0625 / aParticle.GetCyclotronFrequency();
        fFirstStep = false;
    }

    trajmsg_debug("stepsize magnetic moment suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}
void KSTrajControlMagneticMoment::Check(const KSTrajAdiabaticSpinParticle& anInitialParticle,
                                        const KSTrajAdiabaticSpinParticle& aFinalParticle,
                                        const KSTrajAdiabaticSpinError&, bool& aFlag)
{
    double tFinalMagneticMoment = aFinalParticle.GetOrbitalMagneticMoment();
    double tInitialMagneticMoment = anInitialParticle.GetOrbitalMagneticMoment();
    double tAdiabaticityViolation =
        fabs(2. * (tFinalMagneticMoment - tInitialMagneticMoment) / (tFinalMagneticMoment + tInitialMagneticMoment));

    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

    if (tAdiabaticityViolation < fLowerLimit) {
        trajmsg_debug("stepsize magnetic moment increasing stepsize at violation <" << tAdiabaticityViolation << ">"
                                                                                    << eom);
        fTimeStep = 1.5 * fTimeStep;
        aFlag = true;
        return;
    }

    if (tAdiabaticityViolation > fUpperLimit) {
        trajmsg_debug("stepsize magnetic moment decreasing stepsize at violation <" << tAdiabaticityViolation << ">"
                                                                                    << eom);
        fTimeStep = 0.4 * fTimeStep;
        aFlag = false;
        return;
    }

    trajmsg_debug("stepsize magnetic moment keeping stepsize at violation <" << tAdiabaticityViolation << ">" << eom);
    aFlag = true;
    return;
}

void KSTrajControlMagneticMoment::Calculate(const KSTrajAdiabaticParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize magnetic moment on first step" << eom);
        fTimeStep = 0.0625 / aParticle.GetCyclotronFrequency();
        fFirstStep = false;
    }

    trajmsg_debug("stepsize magnetic moment suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}
void KSTrajControlMagneticMoment::Check(const KSTrajAdiabaticParticle& anInitialParticle,
                                        const KSTrajAdiabaticParticle& aFinalParticle, const KSTrajAdiabaticError&,
                                        bool& aFlag)
{
    double tFinalMagneticMoment = aFinalParticle.GetOrbitalMagneticMoment();
    double tInitialMagneticMoment = anInitialParticle.GetOrbitalMagneticMoment();
    double tAdiabaticityViolation =
        fabs(2. * (tFinalMagneticMoment - tInitialMagneticMoment) / (tFinalMagneticMoment + tInitialMagneticMoment));

    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

    if (tAdiabaticityViolation < fLowerLimit) {
        trajmsg_debug("stepsize magnetic moment increasing stepsize at violation <" << tAdiabaticityViolation << ">"
                                                                                    << eom);
        fTimeStep = 1.5 * fTimeStep;
        aFlag = true;
        return;
    }

    if (tAdiabaticityViolation > fUpperLimit) {
        trajmsg_debug("stepsize magnetic moment decreasing stepsize at violation <" << tAdiabaticityViolation << ">"
                                                                                    << eom);
        fTimeStep = 0.4 * fTimeStep;
        aFlag = false;
        return;
    }

    trajmsg_debug("stepsize magnetic moment keeping stepsize at violation <" << tAdiabaticityViolation << ">" << eom);
    aFlag = true;
    return;
}

}  // namespace Kassiopeia
