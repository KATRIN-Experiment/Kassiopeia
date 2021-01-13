#include "KSTrajControlPositionNumericalError.h"

namespace Kassiopeia
{

double KSTrajControlPositionNumericalError::fEpsilon = 1e-14;

KSTrajControlPositionNumericalError::KSTrajControlPositionNumericalError() :
    fAbsoluteError(1e-12),
    fSafetyFactor(0.5),
    fSolverOrder(5),
    fTimeStep(1),
    fFirstStep(true)
{}

KSTrajControlPositionNumericalError::KSTrajControlPositionNumericalError(
    const KSTrajControlPositionNumericalError& aCopy) :
    KSComponent(aCopy),
    fAbsoluteError(aCopy.fAbsoluteError),
    fSafetyFactor(aCopy.fSafetyFactor),
    fSolverOrder(aCopy.fSolverOrder),
    fTimeStep(aCopy.fTimeStep),
    fFirstStep(aCopy.fFirstStep)
{
    if (fSafetyFactor > 1.0) {
        fSafetyFactor = 1.0 / fSafetyFactor;
    };
}

KSTrajControlPositionNumericalError* KSTrajControlPositionNumericalError::Clone() const
{
    return new KSTrajControlPositionNumericalError(*this);
}

KSTrajControlPositionNumericalError::~KSTrajControlPositionNumericalError() = default;

void KSTrajControlPositionNumericalError::ActivateObject()
{
    if (fSafetyFactor > 1.0) {
        fSafetyFactor = 1.0 / fSafetyFactor;
    };
    fFirstStep = true;
    trajmsg_debug("stepsize control <" << GetName() << "> resetting, safety factor is <" << fSafetyFactor << "> "
                                       << eom);
    return;
}

void KSTrajControlPositionNumericalError::Calculate(const KSTrajExactParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize control <" << GetName() << "> on first step" << eom);

        if (aParticle.GetMagneticField().Magnitude() > 1e-9) {
            fTimeStep = 1.0 / aParticle.GetCyclotronFrequency();
        }
        else {
            double v = aParticle.GetVelocity().Magnitude();
            fTimeStep = 100 * (fAbsoluteError / v);
        }
        if (fSafetyFactor > 1.0) {
            fSafetyFactor = 1.0 / fSafetyFactor;
        };
        fFirstStep = false;
    }
    trajmsg_debug("stepsize control <" << GetName() << "> suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}

void KSTrajControlPositionNumericalError::Check(const KSTrajExactParticle& anInitialParticle,
                                                const KSTrajExactParticle& aFinalParticle,
                                                const KSTrajExactError& anError, bool& aFlag)
{
    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

    //get position error
    double position_error_mag = std::fabs((anError.GetPositionError()).Magnitude());
    aFlag = UpdateTimeStep(position_error_mag);
    return;
}

void KSTrajControlPositionNumericalError::Calculate(const KSTrajExactSpinParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize control <" << GetName() << "> on first step" << eom);

        if (aParticle.GetMagneticField().Magnitude() > 1e-9) {
            fTimeStep = 1.0 / aParticle.GetCyclotronFrequency();
        }
        else {
            double v = aParticle.GetVelocity().Magnitude();
            fTimeStep = 100 * (fAbsoluteError / v);
        }
        if (fSafetyFactor > 1.0) {
            fSafetyFactor = 1.0 / fSafetyFactor;
        };
        fFirstStep = false;
    }
    trajmsg_debug("stepsize control <" << GetName() << "> suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}

void KSTrajControlPositionNumericalError::Check(const KSTrajExactSpinParticle& anInitialParticle,
                                                const KSTrajExactSpinParticle& aFinalParticle,
                                                const KSTrajExactSpinError& anError, bool& aFlag)
{
    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

    //get position error
    double position_error_mag = std::fabs((anError.GetPositionError()).Magnitude());
    aFlag = UpdateTimeStep(position_error_mag);
    return;
}

void KSTrajControlPositionNumericalError::Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize control <" << GetName() << "> on first step" << eom);

        if (aParticle.GetMagneticField().Magnitude() > 1e-9) {
            fTimeStep = 1.0 / aParticle.GetCyclotronFrequency();
        }
        else {
            double v = aParticle.GetVelocity().Magnitude();
            fTimeStep = 100 * (fAbsoluteError / v);
        }
        if (fSafetyFactor > 1.0) {
            fSafetyFactor = 1.0 / fSafetyFactor;
        };
        fFirstStep = false;
    }
    trajmsg_debug("stepsize control <" << GetName() << "> suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}

void KSTrajControlPositionNumericalError::Check(const KSTrajAdiabaticSpinParticle& anInitialParticle,
                                                const KSTrajAdiabaticSpinParticle& aFinalParticle,
                                                const KSTrajAdiabaticSpinError& anError, bool& aFlag)
{
    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

    //get position error
    double position_error_mag = std::fabs((anError.GetPositionError()).Magnitude());
    aFlag = UpdateTimeStep(position_error_mag);
    return;
}

void KSTrajControlPositionNumericalError::Calculate(const KSTrajAdiabaticParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize control <" << GetName() << "> on first step" << eom);
        fTimeStep = 1.0 / aParticle.GetCyclotronFrequency();
        if (fSafetyFactor > 1.0) {
            fSafetyFactor = 1.0 / fSafetyFactor;
        };
        fFirstStep = false;
    }
    trajmsg_debug("stepsize control <" << GetName() << "> suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}

void KSTrajControlPositionNumericalError::Check(const KSTrajAdiabaticParticle& anInitialParticle,
                                                const KSTrajAdiabaticParticle& aFinalParticle,
                                                const KSTrajAdiabaticError& anError, bool& aFlag)
{
    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();

    //get position magnitude (on guiding center) and its error
    double position_error_mag = (anError.GetGuidingCenterPositionError()).Magnitude();
    aFlag = UpdateTimeStep(position_error_mag);
    return;
}

bool KSTrajControlPositionNumericalError::UpdateTimeStep(double error)
{
    //check to see if the step was acceptable
    if (error < fAbsoluteError) {
        if (error < fEpsilon * fAbsoluteError) {
            //position error is exceedingly small or zero
            //so we double the stepsize because the normal calculation
            //for estimating the new stepsize may fail
            fTimeStep *= 2;
            trajmsg_debug("stepsize control <" << GetName() << "> doubling stepsize from <" << fTimeStep << "> to <"
                                               << 2 * fTimeStep << "> at position error <" << error << ">" << eom);
            return true;  //aFlag, step succeeded
        }

        //time step is ok, local error does not exceed bounds
        //estimate the next time step
        double beta = error / std::pow(fTimeStep, fSolverOrder);
        double updatedTimeStep = fSafetyFactor * std::pow(fAbsoluteError / beta, 1.0 / fSolverOrder);
        trajmsg_debug("stepsize control <" << GetName() << "> modifying stepsize from <" << fTimeStep << "> to <"
                                           << updatedTimeStep << "> at position error <" << error << ">" << eom);
        fTimeStep = updatedTimeStep;
        return true;  //aFlag, step succeeded
    }
    else {
        //local error is too large so we need to decrease the time step
        //estimate the next time step
        double beta = error / std::pow(fTimeStep, fSolverOrder);
        double updatedTimeStep = fSafetyFactor * std::pow(fAbsoluteError / beta, 1.0 / fSolverOrder);
        trajmsg_debug("stepsize control <" << GetName() << "> decreasing stepsize from <" << fTimeStep << "> to <"
                                           << updatedTimeStep << "> at position error <" << error << ">" << eom);
        fTimeStep = updatedTimeStep;
        return false;  //aFlag, step failed
    }
}


}  // namespace Kassiopeia
