#include "KSTrajControlMomentumNumericalError.h"

namespace Kassiopeia
{

double KSTrajControlMomentumNumericalError::fEpsilon = 1e-14;

KSTrajControlMomentumNumericalError::KSTrajControlMomentumNumericalError() :
    fAbsoluteError(1e-12),
    fSafetyFactor(0.5),
    fSolverOrder(5),
    fTimeStep(1),
    fFirstStep(true)
{}

KSTrajControlMomentumNumericalError::KSTrajControlMomentumNumericalError(
    const KSTrajControlMomentumNumericalError& aCopy) :
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

KSTrajControlMomentumNumericalError* KSTrajControlMomentumNumericalError::Clone() const
{
    return new KSTrajControlMomentumNumericalError(*this);
}

KSTrajControlMomentumNumericalError::~KSTrajControlMomentumNumericalError() = default;

void KSTrajControlMomentumNumericalError::ActivateObject()
{
    if (fSafetyFactor > 1.0) {
        fSafetyFactor = 1.0 / fSafetyFactor;
    };
    fFirstStep = true;
    trajmsg_debug("stepsize momentum numerical error resetting, safety factor is <" << fSafetyFactor << "> " << eom);
    return;
}

void KSTrajControlMomentumNumericalError::Calculate(const KSTrajExactParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize energy on first step" << eom);
        fTimeStep = 1.0 / aParticle.GetCyclotronFrequency();
        if (fSafetyFactor > 1.0) {
            fSafetyFactor = 1.0 / fSafetyFactor;
        };
        fFirstStep = false;
    }
    trajmsg_debug("stepsize numerical error suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}

void KSTrajControlMomentumNumericalError::Check(const KSTrajExactParticle& anInitialParticle,
                                                const KSTrajExactParticle& aFinalParticle,
                                                const KSTrajExactError& anError, bool& aFlag)
{
    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();
    //get Momentum error
    double momentum_error_mag = (anError.GetMomentumError()).Magnitude();
    aFlag = UpdateTimeStep(momentum_error_mag);
    return;
}

void KSTrajControlMomentumNumericalError::Calculate(const KSTrajExactSpinParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize energy on first step" << eom);
        fTimeStep = 1.0 / aParticle.GetCyclotronFrequency();
        if (fSafetyFactor > 1.0) {
            fSafetyFactor = 1.0 / fSafetyFactor;
        };
        fFirstStep = false;
    }
    trajmsg_debug("stepsize numerical error suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}

void KSTrajControlMomentumNumericalError::Check(const KSTrajExactSpinParticle& anInitialParticle,
                                                const KSTrajExactSpinParticle& aFinalParticle,
                                                const KSTrajExactSpinError& anError, bool& aFlag)
{
    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();
    //get Momentum error
    double momentum_error_mag = (anError.GetMomentumError()).Magnitude();
    aFlag = UpdateTimeStep(momentum_error_mag);
    return;
}

void KSTrajControlMomentumNumericalError::Calculate(const KSTrajAdiabaticSpinParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize energy on first step" << eom);
        fTimeStep = 1.0 / aParticle.GetCyclotronFrequency();
        if (fSafetyFactor > 1.0) {
            fSafetyFactor = 1.0 / fSafetyFactor;
        };
        fFirstStep = false;
    }
    trajmsg_debug("stepsize numerical error suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}

void KSTrajControlMomentumNumericalError::Check(const KSTrajAdiabaticSpinParticle& anInitialParticle,
                                                const KSTrajAdiabaticSpinParticle& aFinalParticle,
                                                const KSTrajAdiabaticSpinError& anError, bool& aFlag)
{
    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();
    //get Momentum error
    double momentum_error_mag = (anError.GetMomentumError()).Magnitude();
    aFlag = UpdateTimeStep(momentum_error_mag);
    return;
}

void KSTrajControlMomentumNumericalError::Calculate(const KSTrajAdiabaticParticle& aParticle, double& aValue)
{
    if (fFirstStep == true) {
        trajmsg_debug("stepsize energy on first step" << eom);

        if (aParticle.GetMagneticField().Magnitude() > 1e-9) {
            fTimeStep = 1.0 / aParticle.GetCyclotronFrequency();
        }
        else {
            double p = aParticle.GetMomentum().Magnitude();
            fTimeStep = 100 * (fAbsoluteError / p);
        }

        if (fSafetyFactor > 1.0) {
            fSafetyFactor = 1.0 / fSafetyFactor;
        };
        fFirstStep = false;
    }
    trajmsg_debug("stepsize numerical error suggesting <" << fTimeStep << ">" << eom);
    aValue = fTimeStep;
    return;
}

void KSTrajControlMomentumNumericalError::Check(const KSTrajAdiabaticParticle& anInitialParticle,
                                                const KSTrajAdiabaticParticle& aFinalParticle,
                                                const KSTrajAdiabaticError& anError, bool& aFlag)
{
    fTimeStep = aFinalParticle.GetTime() - anInitialParticle.GetTime();
    //get Momentum magnitude and its error
    double long_error = anError.GetLongitudinalMomentumError();
    double trans_error = anError.GetTransverseMomentumError();
    double momentum_error_mag = std::sqrt(long_error * long_error + trans_error * trans_error);
    aFlag = UpdateTimeStep(momentum_error_mag);
    return;
}

bool KSTrajControlMomentumNumericalError::UpdateTimeStep(double error)
{
    //check to see if the step was acceptable
    if (error < fAbsoluteError) {
        if (error < fEpsilon * fAbsoluteError) {
            //position error is exceedingly small or zero
            //so we double the stepsize because the normal calculation
            //for estimating the new stepsize may fail
            fTimeStep *= 2;
            //                trajmsg_debug( "stepsize position numerical error increasing stepsize from <"<<fTimeStep<<"> to <"<<2*fTimeStep<<"> at position error <" << position_error_mag << ">" << eom) ;
            return true;  //aFlag, step succeeded
        }

        //time step is ok, local error does not exceed bounds
        //estimate the next time step
        double beta = error / std::pow(fTimeStep, fSolverOrder);
        double updatedTimeStep = fSafetyFactor * std::pow(fAbsoluteError / beta, 1.0 / fSolverOrder);
        //            trajmsg_debug( "stepsize position numerical error increasing stepsize from <"<<fTimeStep<<"> to <"<<updatedTimeStep<<"> at position error <" << position_error_mag << ">" << eom) ;
        fTimeStep = updatedTimeStep;
        return true;  //aFlag, step succeeded
    }
    else {
        //local error is too large so we need to decrease the time step
        //estimate the next time step
        double beta = error / std::pow(fTimeStep, fSolverOrder);
        double updatedTimeStep = fSafetyFactor * std::pow(fAbsoluteError / beta, 1.0 / fSolverOrder);
        //            trajmsg_debug( "stepsize position numerical error decreasing stepsize from <"<<fTimeStep<<"> to <"<<updatedTimeStep<<"> at position error <" << position_error_mag << ">" << eom) ;
        fTimeStep = updatedTimeStep;
        return false;  //aFlag, step failed
    }
}


}  // namespace Kassiopeia
