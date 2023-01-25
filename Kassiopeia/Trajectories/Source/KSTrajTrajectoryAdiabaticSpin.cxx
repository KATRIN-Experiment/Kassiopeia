#include "KSTrajTrajectoryAdiabaticSpin.h"

#include "KConst.h"
#include "KSTrajectoriesMessage.h"

#include <limits>
using std::numeric_limits;

using namespace KGeoBag;

using katrin::KThreeVector;

namespace Kassiopeia
{

KSTrajTrajectoryAdiabaticSpin::KSTrajTrajectoryAdiabaticSpin() :
    fInitialParticle(),
    fIntermediateParticle(),
    fFinalParticle(),
    fError(),
    fIntegrator(nullptr),
    fInterpolator(nullptr),
    fTerms(),
    fControls(),
    fPiecewiseTolerance(1e-9),
    fNMaxSegments(1),
    fMaxAttempts(32)
{}
KSTrajTrajectoryAdiabaticSpin::KSTrajTrajectoryAdiabaticSpin(const KSTrajTrajectoryAdiabaticSpin& aCopy) :
    KSComponent(aCopy),
    fInitialParticle(aCopy.fInitialParticle),
    fIntermediateParticle(aCopy.fIntermediateParticle),
    fFinalParticle(aCopy.fFinalParticle),
    fError(aCopy.fError),
    fIntegrator(aCopy.fIntegrator),
    fInterpolator(aCopy.fInterpolator),
    fTerms(aCopy.fTerms),
    fControls(aCopy.fControls),
    fPiecewiseTolerance(aCopy.fPiecewiseTolerance),
    fNMaxSegments(aCopy.fNMaxSegments),
    fMaxAttempts(aCopy.fMaxAttempts)
{}
KSTrajTrajectoryAdiabaticSpin* KSTrajTrajectoryAdiabaticSpin::Clone() const
{
    return new KSTrajTrajectoryAdiabaticSpin(*this);
}
KSTrajTrajectoryAdiabaticSpin::~KSTrajTrajectoryAdiabaticSpin() = default;

void KSTrajTrajectoryAdiabaticSpin::SetIntegrator(KSTrajAdiabaticSpinIntegrator* anIntegrator)
{
    if (fIntegrator == nullptr) {
        fIntegrator = anIntegrator;
        fIntegrator->ClearState();
        return;
    }
    trajmsg(eError) << "cannot set integrator <" << dynamic_cast<katrin::KNamed*>(anIntegrator)->GetName()
                    << "> in <" << this->GetName() << ">" << eom;
    return;
}
void KSTrajTrajectoryAdiabaticSpin::ClearIntegrator(KSTrajAdiabaticSpinIntegrator* anIntegrator)
{
    if (fIntegrator == anIntegrator) {
        fIntegrator = nullptr;
        return;
    }
    trajmsg(eError) << "cannot clear integrator <" << dynamic_cast<katrin::KNamed*>(anIntegrator)->GetName()
                    << "> in <" << this->GetName() << ">" << eom;
    return;
}

void KSTrajTrajectoryAdiabaticSpin::SetInterpolator(KSTrajAdiabaticSpinInterpolator* anInterpolator)
{
    if (fInterpolator == nullptr) {
        fInterpolator = anInterpolator;
        return;
    }
    trajmsg(eError) << "cannot set interpolator <" << dynamic_cast<katrin::KNamed*>(anInterpolator)->GetName()
                    << "> in <" << this->GetName() << ">" << eom;
    return;
}
void KSTrajTrajectoryAdiabaticSpin::ClearInterpolator(KSTrajAdiabaticSpinInterpolator* anInterpolator)
{
    if (fInterpolator == anInterpolator) {
        fInterpolator = nullptr;
        return;
    }
    trajmsg(eError) << "cannot clear interpolator <" << dynamic_cast<katrin::KNamed*>(anInterpolator)->GetName()
                    << "> in <" << this->GetName() << ">" << eom;
    return;
}

void KSTrajTrajectoryAdiabaticSpin::AddTerm(KSTrajAdiabaticSpinDifferentiator* aTerm)
{
    if (fTerms.FindElementByType(aTerm) != -1) {
        trajmsg(eWarning) << "adding term <" << dynamic_cast<katrin::KNamed*>(aTerm)->GetName()
                          << "> to already existing term of same type in <" << this->GetName() << ">" << eom;
    }
    if (fTerms.AddElement(aTerm) != -1) {
        return;
    }
    trajmsg(eError) << "cannot add term <" << dynamic_cast<katrin::KNamed*>(aTerm)->GetName()
                    << "> to <" << this->GetName() << ">" << eom;
    return;
}
void KSTrajTrajectoryAdiabaticSpin::RemoveTerm(KSTrajAdiabaticSpinDifferentiator* aTerm)
{
    if (fTerms.RemoveElement(aTerm) != -1) {
        return;
    }
    trajmsg(eError) << "cannot remove term <" << dynamic_cast<katrin::KNamed*>(aTerm)->GetName()
                    << "> from <" << this->GetName() << ">" << eom;
    return;
}

void KSTrajTrajectoryAdiabaticSpin::AddControl(KSTrajAdiabaticSpinControl* aControl)
{
    if (fControls.FindElementByType(aControl) != -1) {
        trajmsg(eWarning) << "adding control <" << dynamic_cast<katrin::KNamed*>(aControl)->GetName()
                          << "> to already existing control of same type in <" << this->GetName() << ">" << eom;
    }
    if (fControls.AddElement(aControl) != -1) {
        return;
    }
    trajmsg(eError) << "cannot add control <" << dynamic_cast<katrin::KNamed*>(aControl)->GetName()
                    << "> to <" << this->GetName() << ">" << eom;
    return;
}
void KSTrajTrajectoryAdiabaticSpin::RemoveControl(KSTrajAdiabaticSpinControl* aControl)
{
    if (fControls.RemoveElement(aControl) != -1) {
        return;
    }
    trajmsg(eError) << "cannot remove control <" << dynamic_cast<katrin::KNamed*>(aControl)->GetName()
                    << "> from <" << this->GetName() << ">" << eom;
    return;
}

void KSTrajTrajectoryAdiabaticSpin::Reset()
{
    if (fIntegrator != nullptr) {
        fIntegrator->ClearState();
    }
    fInitialParticle = 0.0;
    fFinalParticle = 0.0;
};

void KSTrajTrajectoryAdiabaticSpin::CalculateTrajectory(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                                        KThreeVector& aCenter, double& aRadius, double& aTimeStep)
{
    static const double sMinimalStep = numeric_limits<double>::min();  // smallest positive value

    fInitialParticle = fFinalParticle;
    fInitialParticle.PullFrom(anInitialParticle);
    double currentTime = fInitialParticle.GetTime();

    trajmsg_debug("adiabatic spin trajectory integrating:" << eom);

    bool tFlag = true;
    double tStep;
    double tSmallestStep = numeric_limits<double>::max();
    unsigned int iterCount = 0;

    while (true) {
        for (int tIndex = 0; tIndex < fControls.End(); tIndex++) {
            fControls.ElementAt(tIndex)->Calculate(fInitialParticle, tStep);
            if (tStep < tSmallestStep) {
                tSmallestStep = tStep;
            }
        }

        trajmsg_debug("  time step: <" << tSmallestStep << ">" << eom);

        if (!tFlag) {
            fIntegrator->ClearState();
        };
        fIntegrator->Integrate(currentTime, *this, fInitialParticle, tSmallestStep, fFinalParticle, fError);

        if (fAbortSignal) {
            trajmsg(eWarning) << "trajectory <" << GetName() << "> encountered an abort signal " << eom;
            aFinalParticle = anInitialParticle;
            aFinalParticle.SetActive(false);
            aFinalParticle.SetLabel("trajectory_abort");
            fIntegrator->ClearState();
            break;
        }

        tFlag = true;
        for (int tIndex = 0; tIndex < fControls.End(); tIndex++) {
            fControls.ElementAt(tIndex)->Check(fInitialParticle, fFinalParticle, fError, tFlag);
            if (tFlag == false) {
                double tCurrentStep = tSmallestStep;
                fControls.ElementAt(tIndex)->Calculate(fInitialParticle, tSmallestStep);
                if (fabs(tCurrentStep - tSmallestStep) < sMinimalStep) {
                    trajmsg(eWarning) << "trajectory <" << GetName() << "> could not decide on a valid stepsize" << eom;
                    aFinalParticle = anInitialParticle;
                    aFinalParticle.SetActive(false);
                    aFinalParticle.SetLabel("trajectory_fail");
                    fIntegrator->ClearState();
                    tFlag = true;
                }
                break;
            }
        }

        if (tFlag == true) {
            break;
        }

        if (iterCount > fMaxAttempts) {
            trajmsg(eWarning) << "trajectory <" << GetName()
                              << "> could not perform a sucessful integration step after <" << fMaxAttempts
                              << "> attempts " << eom;
            aFinalParticle = anInitialParticle;
            aFinalParticle.SetActive(false);
            aFinalParticle.SetLabel("trajectory_fail");
            fIntegrator->ClearState();
            tFlag = true;
            break;
        }

        iterCount++;
    }

    fFinalParticle.PushTo(aFinalParticle);
    aFinalParticle.SetLabel(GetName());

    if (fInterpolator != nullptr) {
        fInterpolator->GetPiecewiseLinearApproximation(fPiecewiseTolerance,
                                                       fNMaxSegments,
                                                       fInitialParticle.GetTime(),
                                                       fFinalParticle.GetTime(),
                                                       *fIntegrator,
                                                       *this,
                                                       fInitialParticle,
                                                       fFinalParticle,
                                                       &fIntermediateParticleStates);

        unsigned int n_points = fIntermediateParticleStates.size();
        fBallSupport.Clear();
        //add first and last points before all others
        KThreeVector position = fIntermediateParticleStates[0].GetPosition();
        fBallSupport.AddPoint(KGPoint<3>(position));
        position = fIntermediateParticleStates[n_points - 1].GetPosition();
        fBallSupport.AddPoint(KGPoint<3>(position));

        for (unsigned int i = 1; i < n_points - 1; i++) {
            position = fIntermediateParticleStates[i].GetPosition();
            fBallSupport.AddPoint(KGPoint<3>(position));
        }

        KGBall<3> boundingBall = fBallSupport.GetMinimalBoundingBall();
        aCenter = KThreeVector(boundingBall[0], boundingBall[1], boundingBall[2]);
        aRadius = boundingBall.GetRadius();
        aTimeStep = tSmallestStep;
    }
    else {
        fIntermediateParticleStates.clear();
        fIntermediateParticleStates.push_back(fInitialParticle);
        fIntermediateParticleStates.push_back(fFinalParticle);
        KThreeVector tInitialFinalLine = fFinalParticle.GetPosition() - fInitialParticle.GetPosition();
        aCenter = fInitialParticle.GetPosition() + .5 * tInitialFinalLine;
        aRadius = .5 * tInitialFinalLine.Magnitude();
        aTimeStep = tSmallestStep;
    }

    return;
}

void KSTrajTrajectoryAdiabaticSpin::ExecuteTrajectory(const double& aTimeStep, KSParticle& anIntermediateParticle) const
{
    double currentTime = anIntermediateParticle.GetTime();
    if (fInterpolator != nullptr) {
        fInterpolator->Interpolate(currentTime,
                                   *fIntegrator,
                                   *this,
                                   fInitialParticle,
                                   fFinalParticle,
                                   aTimeStep,
                                   fIntermediateParticle);
        fIntermediateParticle.PushTo(anIntermediateParticle);
        return;
    }
    else {
        fIntegrator->Integrate(currentTime, *this, fInitialParticle, aTimeStep, fIntermediateParticle, fError);
        fIntermediateParticle.PushTo(anIntermediateParticle);
        return;
    }
}

void KSTrajTrajectoryAdiabaticSpin::GetPiecewiseLinearApproximation(
    const KSParticle& anInitialParticle, const KSParticle& /*aFinalParticle*/,
    std::vector<KSParticle>* intermediateParticleStates) const
{
    intermediateParticleStates->clear();
    for (auto& particleState : fIntermediateParticleStates) {
        KSParticle particle(anInitialParticle);
        particleState.PushTo(particle);
        intermediateParticleStates->push_back(particle);
    }
}

void KSTrajTrajectoryAdiabaticSpin::Differentiate(double aTime, const KSTrajAdiabaticSpinParticle& aValue,
                                                  KSTrajAdiabaticSpinDerivative& aDerivative) const
{
    KThreeVector tVelocity = aValue.GetVelocity();

    aDerivative = 0.;
    aDerivative.AddToTime(1.);
    aDerivative.AddToSpeed(tVelocity.Magnitude());

    for (int Index = 0; Index < fTerms.End(); Index++) {
        fTerms.ElementAt(Index)->Differentiate(aTime, aValue, aDerivative);
    }

    return;
}

STATICINT sKSTrajTrajectoryAdiabaticSpinDict =
    KSDictionary<KSTrajTrajectoryAdiabaticSpin>::AddCommand(&KSTrajTrajectoryAdiabaticSpin::SetIntegrator,
                                                            &KSTrajTrajectoryAdiabaticSpin::ClearIntegrator,
                                                            "set_integrator", "clear_integrator") +
    KSDictionary<KSTrajTrajectoryAdiabaticSpin>::AddCommand(&KSTrajTrajectoryAdiabaticSpin::SetInterpolator,
                                                            &KSTrajTrajectoryAdiabaticSpin::ClearInterpolator,
                                                            "set_interpolator", "clear_interpolator") +
    KSDictionary<KSTrajTrajectoryAdiabaticSpin>::AddCommand(&KSTrajTrajectoryAdiabaticSpin::AddTerm,
                                                            &KSTrajTrajectoryAdiabaticSpin::RemoveTerm, "add_term",
                                                            "remove_term") +
    KSDictionary<KSTrajTrajectoryAdiabaticSpin>::AddCommand(&KSTrajTrajectoryAdiabaticSpin::AddControl,
                                                            &KSTrajTrajectoryAdiabaticSpin::RemoveControl,
                                                            "add_control", "remove_control");

}  // namespace Kassiopeia
