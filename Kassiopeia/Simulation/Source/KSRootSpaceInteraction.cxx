#include "KSRootSpaceInteraction.h"

#include "KSException.h"
#include "KSInteractionsMessage.h"
#include "KSParticleFactory.h"

#include <limits>
using std::numeric_limits;

namespace Kassiopeia
{

KSRootSpaceInteraction::KSRootSpaceInteraction() :
    fSpaceInteractions(128),
    fSpaceInteraction(nullptr),
    fStep(nullptr),
    fTerminatorParticle(nullptr),
    fTrajectoryParticle(nullptr),
    fInteractionParticle(nullptr),
    fFinalParticle(nullptr),
    fParticleQueue(nullptr),
    fTrajectory(nullptr)
{}
KSRootSpaceInteraction::KSRootSpaceInteraction(const KSRootSpaceInteraction& aCopy) :
    KSComponent(),
    fSpaceInteractions(aCopy.fSpaceInteractions),
    fSpaceInteraction(aCopy.fSpaceInteraction),
    fStep(aCopy.fStep),
    fTerminatorParticle(aCopy.fTerminatorParticle),
    fTrajectoryParticle(aCopy.fTrajectoryParticle),
    fInteractionParticle(aCopy.fInteractionParticle),
    fFinalParticle(aCopy.fFinalParticle),
    fParticleQueue(aCopy.fParticleQueue),
    fTrajectory(aCopy.fTrajectory)
{}
KSRootSpaceInteraction* KSRootSpaceInteraction::Clone() const
{
    return new KSRootSpaceInteraction(*this);
}
KSRootSpaceInteraction::~KSRootSpaceInteraction() {}

void KSRootSpaceInteraction::CalculateInteraction(const KSTrajectory& aTrajectory,
                                                  const KSParticle& aTrajectoryInitialParticle,
                                                  const KSParticle& aTrajectoryFinalParticle,
                                                  const KThreeVector& aTrajectoryCenter,
                                                  const double& aTrajectoryRadius, const double& aTrajectoryStep,
                                                  KSParticle& anInteractionParticle, double& anInteractionStep,
                                                  bool& anInteractionFlag)
{
    KSParticle tInteractionParticle;
    double tInteractionStep;
    bool tInteractionFlag;

    anInteractionParticle = aTrajectoryFinalParticle;
    anInteractionStep = aTrajectoryStep;
    anInteractionFlag = false;

    try {
        for (int tIndex = 0; tIndex < fSpaceInteractions.End(); tIndex++) {
            fSpaceInteractions.ElementAt(tIndex)->CalculateInteraction(aTrajectory,
                                                                       aTrajectoryInitialParticle,
                                                                       aTrajectoryFinalParticle,
                                                                       aTrajectoryCenter,
                                                                       aTrajectoryRadius,
                                                                       aTrajectoryStep,
                                                                       tInteractionParticle,
                                                                       tInteractionStep,
                                                                       tInteractionFlag);
            if (tInteractionFlag == true) {
                anInteractionFlag = true;
                if (tInteractionStep < anInteractionStep) {
                    anInteractionParticle = tInteractionParticle;
                    fSpaceInteraction = fSpaceInteractions.ElementAt(tIndex);
                }
            }
        }
    }
    catch (KSException const& e) {
        throw KSInteractionError().Nest(e)
            << "Failed to calculate space interaction <" << fSpaceInteraction->GetName() << ">.";
    }
    return;
}
void KSRootSpaceInteraction::ExecuteInteraction(const KSParticle& anInteractionParticle, KSParticle& aFinalParticle,
                                                KSParticleQueue& aSecondaries) const
{
    if (fSpaceInteraction == nullptr) {
        aFinalParticle = anInteractionParticle;
        return;
    }
    try {
        fSpaceInteraction->ExecuteInteraction(anInteractionParticle, aFinalParticle, aSecondaries);
    }
    catch (KSException const& e) {
        throw KSInteractionError().Nest(e)
            << "Failed to execute space interaction <" << fSpaceInteraction->GetName() << ">.";
    }
    return;
}

void KSRootSpaceInteraction::AddSpaceInteraction(KSSpaceInteraction* aSpaceInteraction)
{
    if (fSpaceInteractions.AddElement(aSpaceInteraction) == -1) {
        intmsg(eError) << "<" << GetName() << "> could not add space interaction <" << aSpaceInteraction->GetName()
                       << ">" << eom;
        return;
    }
    intmsg_debug("<" << GetName() << "> adding space interaction <" << aSpaceInteraction->GetName() << ">"
                     << eom) return;
}
void KSRootSpaceInteraction::RemoveSpaceInteraction(KSSpaceInteraction* aSpaceInteraction)
{
    if (fSpaceInteractions.RemoveElement(aSpaceInteraction) == -1) {
        intmsg(eError) << "<" << GetName() << "> could not remove space interaction <" << aSpaceInteraction->GetName()
                       << ">" << eom;
        return;
    }
    intmsg_debug("<" << GetName() << "> removing space interaction <" << aSpaceInteraction->GetName() << ">"
                     << eom) return;
}

void KSRootSpaceInteraction::SetStep(KSStep* aStep)
{
    fStep = aStep;
    fTerminatorParticle = &(aStep->TerminatorParticle());
    fTrajectoryParticle = &(aStep->TrajectoryParticle());
    fInteractionParticle = &(aStep->InteractionParticle());
    fFinalParticle = &(aStep->FinalParticle());
    fParticleQueue = &(aStep->ParticleQueue());
    return;
}
void KSRootSpaceInteraction::SetTrajectory(KSTrajectory* aTrajectory)
{
    fTrajectory = aTrajectory;
    return;
}

void KSRootSpaceInteraction::CalculateInteraction()
{
    *fInteractionParticle = *fTrajectoryParticle;

    if (fSpaceInteractions.End() == 0) {
        intmsg_debug("space interaction calculation:" << eom) intmsg_debug("  no space interactions active" << eom)
            intmsg_debug("  interaction name: <" << fStep->GetSpaceInteractionName() << ">" << eom)
                intmsg_debug("  interaction step: <" << fStep->GetSpaceInteractionStep() << ">" << eom) intmsg_debug(
                    "  interaction flag: <" << fStep->GetSpaceInteractionFlag() << ">" << eom)

                    intmsg_debug("space interaction calculation interaction particle state: " << eom) intmsg_debug(
                        "  final particle space: <"
                        << (fInteractionParticle->GetCurrentSpace() ? fInteractionParticle->GetCurrentSpace()->GetName()
                                                                    : "")
                        << ">" << eom) intmsg_debug("  final particle surface: <"
                                                    << (fInteractionParticle->GetCurrentSurface()
                                                            ? fInteractionParticle->GetCurrentSurface()->GetName()
                                                            : "")
                                                    << ">" << eom) intmsg_debug("  final particle time: <"
                                                                                << fInteractionParticle->GetTime()
                                                                                << ">" << eom)
                        intmsg_debug("  final particle length: <" << fInteractionParticle->GetLength() << ">" << eom)
                            intmsg_debug("  final particle position: <"
                                         << fInteractionParticle->GetPosition().X() << ", "
                                         << fInteractionParticle->GetPosition().Y() << ", "
                                         << fInteractionParticle->GetPosition().Z() << ">" << eom)
                                intmsg_debug("  final particle momentum: <"
                                             << fInteractionParticle->GetMomentum().X() << ", "
                                             << fInteractionParticle->GetMomentum().Y() << ", "
                                             << fInteractionParticle->GetMomentum().Z() << ">" << eom)
                                    intmsg_debug("  final particle kinetic energy: <"
                                                 << fInteractionParticle->GetKineticEnergy_eV() << ">" << eom)
                                        intmsg_debug("  final particle electric field: <"
                                                     << fInteractionParticle->GetElectricField().X() << ","
                                                     << fInteractionParticle->GetElectricField().Y() << ","
                                                     << fInteractionParticle->GetElectricField().Z() << ">" << eom)
                                            intmsg_debug("  final particle magnetic field: <"
                                                         << fInteractionParticle->GetMagneticField().X() << ","
                                                         << fInteractionParticle->GetMagneticField().Y() << ","
                                                         << fInteractionParticle->GetMagneticField().Z() << ">" << eom)
                                                intmsg_debug("  final particle angle to magnetic field: <"
                                                             << fInteractionParticle->GetPolarAngleToB() << ">" << eom)
                                                    intmsg_debug("  final particle spin: "
                                                                 << fInteractionParticle->GetSpin() << eom)
                                                        intmsg_debug("  final particle spin0: <"
                                                                     << fInteractionParticle->GetSpin0() << ">" << eom)
                                                            intmsg_debug("  final particle aligned spin: <"
                                                                         << fInteractionParticle->GetAlignedSpin()
                                                                         << ">" << eom)
                                                                intmsg_debug("  final particle spin angle: <"
                                                                             << fInteractionParticle->GetSpinAngle()
                                                                             << ">" << eom)

                                                                    return;
    }

    CalculateInteraction(*fTrajectory,
                         *fTerminatorParticle,
                         *fTrajectoryParticle,
                         fStep->TrajectoryCenter(),
                         fStep->TrajectoryRadius(),
                         fStep->TrajectoryStep(),
                         *fInteractionParticle,
                         fStep->SpaceInteractionStep(),
                         fStep->SpaceInteractionFlag());

    if (fStep->SpaceInteractionFlag() == true) {
        intmsg_debug("space interaction calculation:" << eom) intmsg_debug("  space interaction may occur" << eom)
    }
    else {
        intmsg_debug("space interaction calculation:" << eom) intmsg_debug("  space interaction will not occur" << eom)
    }

    intmsg_debug("space interaction calculation interaction particle state: " << eom) intmsg_debug(
        "  interaction particle space: <"
        << (fInteractionParticle->GetCurrentSpace() ? fInteractionParticle->GetCurrentSpace()->GetName() : "") << ">"
        << eom)
        intmsg_debug(
            "  interaction particle surface: <"
            << (fInteractionParticle->GetCurrentSurface() ? fInteractionParticle->GetCurrentSurface()->GetName() : "")
            << ">" << eom)
            intmsg_debug("  interaction particle time: <" << fInteractionParticle->GetTime() << ">" << eom)
                intmsg_debug("  interaction particle length: <" << fInteractionParticle->GetLength() << ">" << eom)
                    intmsg_debug("  interaction particle position: <" << fInteractionParticle->GetPosition().X() << ", "
                                                                      << fInteractionParticle->GetPosition().Y() << ", "
                                                                      << fInteractionParticle->GetPosition().Z() << ">"
                                                                      << eom)
                        intmsg_debug("  interaction particle momentum: <"
                                     << fInteractionParticle->GetMomentum().X() << ", "
                                     << fInteractionParticle->GetMomentum().Y() << ", "
                                     << fInteractionParticle->GetMomentum().Z() << ">" << eom)
                            intmsg_debug("  interaction particle kinetic energy: <"
                                         << fInteractionParticle->GetKineticEnergy_eV() << ">" << eom)
                                intmsg_debug("  interaction particle electric field: <"
                                             << fInteractionParticle->GetElectricField().X() << ","
                                             << fInteractionParticle->GetElectricField().Y() << ","
                                             << fInteractionParticle->GetElectricField().Z() << ">" << eom)
                                    intmsg_debug("  interaction particle magnetic field: <"
                                                 << fInteractionParticle->GetMagneticField().X() << ","
                                                 << fInteractionParticle->GetMagneticField().Y() << ","
                                                 << fInteractionParticle->GetMagneticField().Z() << ">" << eom)
                                        intmsg_debug("  interaction particle angle to magnetic field: <"
                                                     << fInteractionParticle->GetPolarAngleToB() << ">" << eom);
    intmsg_debug("  interaction particle spin: " << fInteractionParticle->GetSpin() << eom)
        intmsg_debug("  interaction particle spin0: <" << fInteractionParticle->GetSpin0() << ">" << eom) intmsg_debug(
            "  interaction particle aligned spin: <" << fInteractionParticle->GetAlignedSpin() << ">" << eom)
            intmsg_debug("  interaction particle spin angle: <" << fInteractionParticle->GetSpinAngle() << ">" << eom)

                return;
}

void KSRootSpaceInteraction::ExecuteInteraction()
{
    ExecuteInteraction(*fInteractionParticle, *fFinalParticle, *fParticleQueue);
    fFinalParticle->ReleaseLabel(fStep->SpaceInteractionName());

    fStep->ContinuousTime() = fInteractionParticle->GetTime() - fTerminatorParticle->GetTime();
    fStep->ContinuousLength() = fInteractionParticle->GetLength() - fTerminatorParticle->GetLength();
    fStep->ContinuousEnergyChange() =
        fInteractionParticle->GetKineticEnergy_eV() - fTerminatorParticle->GetKineticEnergy_eV();
    fStep->ContinuousMomentumChange() =
        (fInteractionParticle->GetMomentum() - fTerminatorParticle->GetMomentum()).Magnitude();
    fStep->DiscreteSecondaries() = fParticleQueue->size();
    fStep->DiscreteEnergyChange() = fFinalParticle->GetKineticEnergy_eV() - fInteractionParticle->GetKineticEnergy_eV();
    fStep->DiscreteMomentumChange() = (fFinalParticle->GetMomentum() - fInteractionParticle->GetMomentum()).Magnitude();

    intmsg_debug("space interaction execution:" << eom) intmsg_debug("  space interaction name: <"
                                                                     << fStep->SpaceInteractionName() << ">" << eom)
        intmsg_debug("  step continuous time: <" << fStep->ContinuousTime() << ">" << eom)
            intmsg_debug("  step continuous length: <" << fStep->ContinuousLength() << ">" << eom) intmsg_debug(
                "  step continuous energy change: <" << fStep->ContinuousEnergyChange() << ">" << eom)
                intmsg_debug("  step continuous momentum change: <" << fStep->ContinuousMomentumChange() << ">" << eom)
                    intmsg_debug("  step discrete secondaries: <" << fStep->DiscreteSecondaries() << ">" << eom)
                        intmsg_debug("  step discrete energy change: <" << fStep->DiscreteEnergyChange() << ">" << eom)
                            intmsg_debug("  step discrete momentum change: <" << fStep->DiscreteMomentumChange() << ">"
                                                                              << eom);

    intmsg_debug("space interaction execution final particle state: " << eom)
        intmsg_debug("  final particle space: <"
                     << (fFinalParticle->GetCurrentSpace() ? fFinalParticle->GetCurrentSpace()->GetName() : "") << ">"
                     << eom)
            intmsg_debug("  final particle surface: <"
                         << (fFinalParticle->GetCurrentSurface() ? fFinalParticle->GetCurrentSurface()->GetName() : "")
                         << ">"
                         << eom) intmsg_debug("  final particle time: <" << fFinalParticle->GetTime() << ">" << eom)
                intmsg_debug("  final particle length: <" << fFinalParticle->GetLength() << ">" << eom)
                    intmsg_debug("  final particle position: <" << fFinalParticle->GetPosition().X() << ", "
                                                                << fFinalParticle->GetPosition().Y() << ", "
                                                                << fFinalParticle->GetPosition().Z() << ">" << eom)
                        intmsg_debug("  final particle momentum: <" << fFinalParticle->GetMomentum().X() << ", "
                                                                    << fFinalParticle->GetMomentum().Y() << ", "
                                                                    << fFinalParticle->GetMomentum().Z() << ">" << eom)
                            intmsg_debug("  final particle kinetic energy: <" << fFinalParticle->GetKineticEnergy_eV()
                                                                              << ">" << eom)
                                intmsg_debug("  final particle electric field: <"
                                             << fFinalParticle->GetElectricField().X() << ","
                                             << fFinalParticle->GetElectricField().Y() << ","
                                             << fFinalParticle->GetElectricField().Z() << ">" << eom)
                                    intmsg_debug("  final particle magnetic field: <"
                                                 << fFinalParticle->GetMagneticField().X() << ","
                                                 << fFinalParticle->GetMagneticField().Y() << ","
                                                 << fFinalParticle->GetMagneticField().Z() << ">" << eom)
                                        intmsg_debug("  final particle angle to magnetic field: <"
                                                     << fFinalParticle->GetPolarAngleToB() << ">" << eom);
    intmsg_debug("  final particle spin: " << fFinalParticle->GetSpin() << eom)
        intmsg_debug("  final particle spin0: <" << fFinalParticle->GetSpin0() << ">" << eom)
            intmsg_debug("  final particle aligned spin: <" << fFinalParticle->GetAlignedSpin() << ">" << eom)
                intmsg_debug("  final particle spin angle: <" << fFinalParticle->GetSpinAngle() << ">" << eom)

                    return;
}

void KSRootSpaceInteraction::PushUpdateComponent()
{
    for (int tIndex = 0; tIndex < fSpaceInteractions.End(); tIndex++) {
        fSpaceInteractions.ElementAt(tIndex)->PushUpdate();
    }
}
void KSRootSpaceInteraction::PushDeupdateComponent()
{
    for (int tIndex = 0; tIndex < fSpaceInteractions.End(); tIndex++) {
        fSpaceInteractions.ElementAt(tIndex)->PushDeupdate();
    }
}

STATICINT sKSRootSpaceInteractionDict = KSDictionary<KSRootSpaceInteraction>::AddCommand(
    &KSRootSpaceInteraction::AddSpaceInteraction, &KSRootSpaceInteraction::RemoveSpaceInteraction,
    "add_space_interaction", "remove_space_interaction");

}  // namespace Kassiopeia
