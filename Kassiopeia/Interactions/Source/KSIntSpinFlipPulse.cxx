#include "KSIntSpinFlipPulse.h"

#include "KRandom.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

using katrin::KThreeVector;

namespace Kassiopeia
{

KSIntSpinFlipPulse::KSIntSpinFlipPulse() : fDone(false), fTime(0.) {}

KSIntSpinFlipPulse::KSIntSpinFlipPulse(const KSIntSpinFlipPulse& aCopy) :
    KSComponent(aCopy),
    KSComponentTemplate<KSIntSpinFlipPulse, KSSpaceInteraction>(aCopy),
    fDone(aCopy.fDone),
    fTime(aCopy.fTime)
{}

KSIntSpinFlipPulse* KSIntSpinFlipPulse::Clone() const
{
    return new KSIntSpinFlipPulse(*this);
}

KSIntSpinFlipPulse::~KSIntSpinFlipPulse() = default;


void KSIntSpinFlipPulse::CalculateInteraction(const KSTrajectory& /*aTrajectory*/,
                                              const KSParticle& /*aTrajectoryInitialParticle*/,
                                              const KSParticle& aTrajectoryFinalParticle,
                                              const KThreeVector& /*aTrajectoryCenter*/,
                                              const double& /*aTrajectoryRadius*/,
                                              const double& /*aTrajectoryTimeStep*/, KSParticle& anInteractionParticle,
                                              double& /*aTimeStep*/, bool& aFlag)
{
    anInteractionParticle = aTrajectoryFinalParticle;

    // For some reason ths stops the interaction from working, and removing
    // it doesn't seem to break anything.
    // aTimeStep = aTrajectoryTimeStep;

    if (aTrajectoryFinalParticle.GetTime() < fTime) {
        aFlag = false;
        fDone = false;
    }
    else if (fDone) {
        aFlag = false;
    }
    else {
        aFlag = true;
        // fDone = true;  Due to double precision errors in the time, this should occur on execution instead
    }

    return;
}

void KSIntSpinFlipPulse::ExecuteInteraction(const KSParticle& anInteractionParticle, KSParticle& aFinalParticle,
                                            KSParticleQueue& /*aSecondaries*/) const
{

    fDone = true;

    aFinalParticle = anInteractionParticle;
    aFinalParticle.SetAlignedSpin(-1. * aFinalParticle.GetAlignedSpin());
    aFinalParticle.SetSpinAngle(katrin::KConst::Pi() + aFinalParticle.GetSpinAngle());
    aFinalParticle.RecalculateSpinGlobal();

    return;
}

void KSIntSpinFlipPulse::SetTime(const double& aTime)
{
    fTime = aTime;
    return;
}

}  // namespace Kassiopeia
