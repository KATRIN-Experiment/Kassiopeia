#include "KSTrajTrajectoryAdiabatic.h"

#include "KConst.h"
#include "KSTrajectoriesMessage.h"

#include <limits>
using std::numeric_limits;

using namespace KGeoBag;

namespace Kassiopeia
{

KSTrajTrajectoryAdiabatic::KSTrajTrajectoryAdiabatic() :
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
    fUseTruePostion(true),
    fCyclotronFraction(1.0 / 8.0),
    fMaxAttempts(32)
{}
KSTrajTrajectoryAdiabatic::KSTrajTrajectoryAdiabatic(const KSTrajTrajectoryAdiabatic& aCopy) :
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
    fUseTruePostion(aCopy.fUseTruePostion),
    fCyclotronFraction(aCopy.fCyclotronFraction),
    fMaxAttempts(aCopy.fMaxAttempts)
{}
KSTrajTrajectoryAdiabatic* KSTrajTrajectoryAdiabatic::Clone() const
{
    return new KSTrajTrajectoryAdiabatic(*this);
}
KSTrajTrajectoryAdiabatic::~KSTrajTrajectoryAdiabatic() = default;

void KSTrajTrajectoryAdiabatic::SetIntegrator(KSTrajAdiabaticIntegrator* anIntegrator)
{
    if (fIntegrator == nullptr) {
        fIntegrator = anIntegrator;
        return;
    }
    trajmsg(eError) << "cannot set integrator in <" << this->GetName() << "> with <" << anIntegrator << ">" << eom;
    return;
}
void KSTrajTrajectoryAdiabatic::ClearIntegrator(KSTrajAdiabaticIntegrator* anIntegrator)
{
    if (fIntegrator == anIntegrator) {
        fIntegrator = nullptr;
        return;
    }
    trajmsg(eError) << "cannot clear integrator in <" << this->GetName() << "> with <" << anIntegrator << ">" << eom;
    return;
}

void KSTrajTrajectoryAdiabatic::SetInterpolator(KSTrajAdiabaticInterpolator* anInterpolator)
{
    if (fInterpolator == nullptr) {
        fInterpolator = anInterpolator;
        return;
    }
    trajmsg(eError) << "cannot set interpolator in <" << this->GetName() << "> with <" << anInterpolator << ">" << eom;
    return;
}
void KSTrajTrajectoryAdiabatic::ClearInterpolator(KSTrajAdiabaticInterpolator* anInterpolator)
{
    if (fInterpolator == anInterpolator) {
        fInterpolator = nullptr;
        return;
    }
    trajmsg(eError) << "cannot clear interpolator in <" << this->GetName() << "> with <" << anInterpolator << ">"
                    << eom;
    return;
}

void KSTrajTrajectoryAdiabatic::AddTerm(KSTrajAdiabaticDifferentiator* aTerm)
{
    if (fTerms.AddElement(aTerm) != -1) {
        return;
    }
    trajmsg(eError) << "cannot add term <" << aTerm << "> to <" << this->GetName() << ">" << eom;
    return;
}
void KSTrajTrajectoryAdiabatic::RemoveTerm(KSTrajAdiabaticDifferentiator* aTerm)
{
    if (fTerms.RemoveElement(aTerm) != -1) {
        return;
    }
    trajmsg(eError) << "cannot remove term <" << aTerm << "> from <" << this->GetName() << ">" << eom;
    return;
}

void KSTrajTrajectoryAdiabatic::AddControl(KSTrajAdiabaticControl* aControl)
{
    if (fControls.AddElement(aControl) != -1) {
        return;
    }
    trajmsg(eError) << "cannot add step <" << aControl << "> to <" << this->GetName() << ">" << eom;
    return;
}
void KSTrajTrajectoryAdiabatic::RemoveControl(KSTrajAdiabaticControl* aControl)
{
    if (fControls.RemoveElement(aControl) != -1) {
        return;
    }
    trajmsg(eError) << "cannot remove step <" << aControl << "> from <" << this->GetName() << ">" << eom;
    return;
}

void KSTrajTrajectoryAdiabatic::Reset()
{
    if (fIntegrator != nullptr) {
        fIntegrator->ClearState();
    }
    fInitialParticle = 0.0;
    fFinalParticle = 0.0;
};

void KSTrajTrajectoryAdiabatic::CalculateTrajectory(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                                    KThreeVector& aCenter, double& aRadius, double& aTimeStep)
{
    static const double sMinimalStep = numeric_limits<double>::min();  // smallest positive value

    fInitialParticle = fFinalParticle;
    fInitialParticle.PullFrom(anInitialParticle);
    double currentTime = fInitialParticle.GetTime();

    //spray
    trajmsg_debug("initial real position: " << fInitialParticle.GetPosition() << ret)
        trajmsg_debug("initial real momentum: " << fInitialParticle.GetMomentum() << ret)
            trajmsg_debug("initial gc position: " << fInitialParticle.GetGuidingCenter() << ret)
                trajmsg_debug("initial gc alpha: " << fInitialParticle.GetAlpha() << ret)
                    trajmsg_debug("initial gc beta: " << fInitialParticle.GetBeta() << ret)
                        trajmsg_debug("initial parallel momentum: <" << fInitialParticle[5] << ">" << ret)
                            trajmsg_debug("initial perpendicular momentum: <" << fInitialParticle[6] << ">" << ret)
                                trajmsg_debug("initial kinetic energy is: <"
                                              << fInitialParticle.GetKineticEnergy() / katrin::KConst::Q() << ">"
                                              << eom)

                                    bool tFlag;
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

        trajmsg_debug("time step is <" << tSmallestStep << ">" << eom);

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

                trajmsg_debug("trajectory <" << GetName()
                                             << "> re-attempting integration step after stepsize control check failed "
                                             << eom)

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
            break;
        }

        iterCount++;
    }

    //compute rotation minimizing frame via double-reflection
    const KThreeVector tInitialPosition = fInitialParticle.GetGuidingCenter();
    const KThreeVector tInitialTangent = fInitialParticle.GetMagneticField().Unit();
    const KThreeVector tInitialNormal = fInitialParticle.GetAlpha();
    const KThreeVector tFinalPosition = fFinalParticle.GetGuidingCenter();
    const KThreeVector tFinalTangent = fFinalParticle.GetMagneticField().Unit();

    KThreeVector tReflectionOneVector = tFinalPosition - tInitialPosition;
    double tReflectionOne = tReflectionOneVector.MagnitudeSquared();
    KThreeVector tTangentA =
        tInitialTangent - (2. / tReflectionOne) * (tReflectionOneVector.Dot(tInitialTangent)) * tReflectionOneVector;
    KThreeVector tNormalA =
        tInitialNormal - (2. / tReflectionOne) * (tReflectionOneVector.Dot(tInitialNormal)) * tReflectionOneVector;

    KThreeVector tReflectionTwoVector = tFinalTangent - tTangentA;
    double tReflectionTwo = tReflectionTwoVector.MagnitudeSquared();
    KThreeVector tNormalB =
        tNormalA - (2. / tReflectionTwo) * (tReflectionTwoVector.Dot(tNormalA)) * tReflectionTwoVector;
    KThreeVector tNormalC = tNormalB - tNormalB.Dot(tFinalTangent) * tFinalTangent;
    KThreeVector tFinalNormal = tNormalC.Unit();
    KThreeVector tFinalBinormal = tFinalTangent.Cross(tFinalNormal).Unit();

    fFinalParticle.SetAlpha(tFinalNormal);
    fFinalParticle.SetBeta(tFinalBinormal);
    fFinalParticle.PushTo(aFinalParticle);
    aFinalParticle.SetLabel(GetName());

    trajmsg_debug("final real position: " << fFinalParticle.GetPosition() << ret)
        trajmsg_debug("final real momentum: " << fFinalParticle.GetMomentum() << ret)
            trajmsg_debug("final gc position: " << fFinalParticle.GetGuidingCenter() << ret)
                trajmsg_debug("final gc alpha: " << fFinalParticle.GetAlpha() << ret)
                    trajmsg_debug("final gc beta: " << fFinalParticle.GetBeta() << ret)
                        trajmsg_debug("final parallel momentum: <" << fFinalParticle[5] << ">" << ret)
                            trajmsg_debug("final perpendicular momentum: <" << fFinalParticle[6] << ">" << ret)
                                trajmsg_debug("final kinetic energy is: <"
                                              << fFinalParticle.GetKineticEnergy() / katrin::KConst::Q() << ">" << eom)

                                    if (fInterpolator != nullptr)
    {
        if (fUseTruePostion) {
            //we want to approximate the path of the true particle (not just the guiding center)
            //using the sampling size fCyclotronFraction and the mean cyclotron frequency to dictate
            //the time step between states and the number of intermediate states we need
            double mean_cyclotron =
                (fInitialParticle.GetCyclotronFrequency() + fFinalParticle.GetCyclotronFrequency()) / 2.0;
            double n_periods = (fFinalParticle.GetTime() - fInitialParticle.GetTime()) * mean_cyclotron;
            auto n_segments = static_cast<unsigned int>(n_periods / fCyclotronFraction);
            if (n_segments < 1) {
                n_segments = 1;
            };

            fInterpolator->GetFixedPiecewiseLinearApproximation(n_segments,
                                                                fInitialParticle.GetTime(),
                                                                fFinalParticle.GetTime(),
                                                                *fIntegrator,
                                                                *this,
                                                                fInitialParticle,
                                                                fFinalParticle,
                                                                &fIntermediateParticleStates);
        }
        else {
            //only need to approximate position
            //(number of samples depends on smoothness of guiding center path)
            fInterpolator->GetPiecewiseLinearApproximation(fPiecewiseTolerance,
                                                           fNMaxSegments,
                                                           fInitialParticle.GetTime(),
                                                           fFinalParticle.GetTime(),
                                                           *fIntegrator,
                                                           *this,
                                                           fInitialParticle,
                                                           fFinalParticle,
                                                           &fIntermediateParticleStates);
        }

        //compute the bounding ball
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
    else
    {
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

void KSTrajTrajectoryAdiabatic::ExecuteTrajectory(const double& aTimeStep, KSParticle& anIntermediateParticle) const
{
    double currentTime = anIntermediateParticle.GetTime();
    if (fInterpolator) {
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
        trajmsg_debug("execute trajectory without interpolation: " << ret)
            trajmsg_debug("timestep: " << aTimeStep << ret)
                trajmsg_debug("initial real position: " << fInitialParticle.GetPosition() << ret)
                    trajmsg_debug("initial real momentum: " << fInitialParticle.GetMomentum() << ret)
                        trajmsg_debug("initial gc position: " << fInitialParticle.GetGuidingCenter() << ret)
                            trajmsg_debug("initial gc alpha: " << fInitialParticle.GetAlpha() << ret)
                                trajmsg_debug("initial gc beta: " << fInitialParticle.GetBeta() << ret)
                                    trajmsg_debug("initial parallel momentum: <" << fInitialParticle[5] << ">" << ret)
                                        trajmsg_debug("initial perpendicular momentum: <" << fInitialParticle[6] << ">"
                                                                                          << ret)
                                            trajmsg_debug("initial kinetic energy is: <"
                                                          << fInitialParticle.GetKineticEnergy() / katrin::KConst::Q()
                                                          << ">" << eom)

                                                if (aTimeStep == 0.0)
        {
            fIntermediateParticle = fInitialParticle;
            trajmsg_debug("timestep was 0, using initial particle" << eom)
                fIntermediateParticle.PushTo(anIntermediateParticle);
            return;
        }

        fIntegrator->Integrate(currentTime, *this, fInitialParticle, aTimeStep, fIntermediateParticle, fError);

        //compute rotation minimizing frame via double-reflection
        const KThreeVector tInitialPosition = fInitialParticle.GetGuidingCenter();
        const KThreeVector tInitialTangent = fInitialParticle.GetMagneticField().Unit();
        const KThreeVector tInitialNormal = fInitialParticle.GetAlpha();
        const KThreeVector tFinalPosition = fIntermediateParticle.GetGuidingCenter();
        const KThreeVector tFinalTangent = fIntermediateParticle.GetMagneticField().Unit();

        KThreeVector tReflectionOneVector = tFinalPosition - tInitialPosition;
        double tReflectionOne = tReflectionOneVector.MagnitudeSquared();
        KThreeVector tTangentA = tInitialTangent - (2. / tReflectionOne) * (tReflectionOneVector.Dot(tInitialTangent)) *
                                                       tReflectionOneVector;
        KThreeVector tNormalA =
            tInitialNormal - (2. / tReflectionOne) * (tReflectionOneVector.Dot(tInitialNormal)) * tReflectionOneVector;

        KThreeVector tReflectionTwoVector = tFinalTangent - tTangentA;
        double tReflectionTwo = tReflectionTwoVector.MagnitudeSquared();
        KThreeVector tNormalB =
            tNormalA - (2. / tReflectionTwo) * (tReflectionTwoVector.Dot(tNormalA)) * tReflectionTwoVector;
        KThreeVector tNormalC = tNormalB - tNormalB.Dot(tFinalTangent) * tFinalTangent;
        KThreeVector tFinalNormal = tNormalC.Unit();
        KThreeVector tFinalBinormal = tFinalTangent.Cross(tFinalNormal).Unit();

        fIntermediateParticle.SetAlpha(tFinalNormal);
        fIntermediateParticle.SetBeta(tFinalBinormal);

        trajmsg_debug("intermediate real position: " << fIntermediateParticle.GetPosition() << ret)
            trajmsg_debug("intermediate real momentum: " << fIntermediateParticle.GetMomentum() << ret)
                trajmsg_debug("intermediate gc position: " << fIntermediateParticle.GetGuidingCenter() << ret)
                    trajmsg_debug("intermediate gc alpha: " << fIntermediateParticle.GetAlpha() << ret)
                        trajmsg_debug("intermediate gc beta: " << fIntermediateParticle.GetBeta() << ret)
                            trajmsg_debug("intermediate parallel momentum: <" << fIntermediateParticle[5] << ">" << ret)
                                trajmsg_debug("intermediate perpendicular momentum: <" << fIntermediateParticle[6]
                                                                                       << ">" << ret)
                                    trajmsg_debug("intermediate kinetic energy is: <"
                                                  << fIntermediateParticle.GetKineticEnergy() / katrin::KConst::Q()
                                                  << ">" << eom)

                                        fIntermediateParticle.PushTo(anIntermediateParticle);
        fFinalParticle = fIntermediateParticle;
        return;
    }
}

void KSTrajTrajectoryAdiabatic::GetPiecewiseLinearApproximation(
    const KSParticle& anInitialParticle, const KSParticle& /*aFinalParticle*/,
    std::vector<KSParticle>* intermediateParticleStates) const
{
    intermediateParticleStates->clear();
    for (auto& particleState : fIntermediateParticleStates) {
        KSParticle particle(anInitialParticle);
        particle.ResetFieldCaching();
        particleState.PushTo(particle);
        intermediateParticleStates->push_back(particle);
    }
}

void KSTrajTrajectoryAdiabatic::Differentiate(double aTime, const KSTrajAdiabaticParticle& aValue,
                                              KSTrajAdiabaticDerivative& aDerivative) const
{
    //force the cached calculation of magnetic field and gradient combined
    //(otherwise the magnetic field will be calculated single here and the gradient later also
    aValue.GetMagneticFieldAndGradient();

    //the same for the combined cached calculation of the electric field and potential
    aValue.GetElectricFieldAndPotential();

    double tLongVelocity = aValue.GetLongVelocity();
    double tTransVelocity = aValue.GetTransVelocity();

    // trajmsg_debug( "traj adiabatic long momentum = "<<  aValue.GetLongMomentum() << eom)
    // trajmsg_debug( "traj adiabatic trans momentum = "<< aValue.GetTransMomentum() << eom)
    // trajmsg_debug( "traj adiabatic lorentz factor = "<< aValue.GetLorentzFactor() << eom)
    // trajmsg_debug( "traj adiabatic long velocity = "<< tLongVelocity << eom)
    // trajmsg_debug( "traj adiabatic trans velocity = "<< tLongVelocity << eom)

    aDerivative = 0.;
    aDerivative.AddToTime(1.);
    aDerivative.AddToSpeed(sqrt(tLongVelocity * tLongVelocity + tTransVelocity * tTransVelocity));

    for (int Index = 0; Index < fTerms.End(); Index++) {
        fTerms.ElementAt(Index)->Differentiate(aTime, aValue, aDerivative);
    }

    return;
}

STATICINT sKSTrajTrajectoryAdiabaticDict =
    KSDictionary<KSTrajTrajectoryAdiabatic>::AddCommand(&KSTrajTrajectoryAdiabatic::SetIntegrator,
                                                        &KSTrajTrajectoryAdiabatic::ClearIntegrator, "set_integrator",
                                                        "clear_integrator") +
    KSDictionary<KSTrajTrajectoryAdiabatic>::AddCommand(&KSTrajTrajectoryAdiabatic::SetInterpolator,
                                                        &KSTrajTrajectoryAdiabatic::ClearInterpolator,
                                                        "set_interpolator", "clear_interpolator") +
    KSDictionary<KSTrajTrajectoryAdiabatic>::AddCommand(
        &KSTrajTrajectoryAdiabatic::AddTerm, &KSTrajTrajectoryAdiabatic::RemoveTerm, "add_term", "remove_term") +
    KSDictionary<KSTrajTrajectoryAdiabatic>::AddCommand(&KSTrajTrajectoryAdiabatic::AddControl,
                                                        &KSTrajTrajectoryAdiabatic::RemoveControl, "add_control",
                                                        "remove_control");

}  // namespace Kassiopeia
