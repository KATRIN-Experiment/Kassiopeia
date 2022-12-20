#include "KSRootSurfaceInteraction.h"

#include "KRandom.h"
#include "KSException.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

namespace Kassiopeia
{

KSRootSurfaceInteraction::KSRootSurfaceInteraction() :
    fSurfaceInteraction(nullptr),
    fStep(nullptr),
    fTerminatorParticle(nullptr),
    fInteractionParticle(nullptr),
    fParticleQueue(nullptr)
{}
KSRootSurfaceInteraction::KSRootSurfaceInteraction(const KSRootSurfaceInteraction& aCopy) :
    KSComponent(aCopy),
    fSurfaceInteraction(aCopy.fSurfaceInteraction),
    fStep(aCopy.fStep),
    fTerminatorParticle(aCopy.fTerminatorParticle),
    fInteractionParticle(aCopy.fInteractionParticle),
    fParticleQueue(aCopy.fParticleQueue)
{}
KSRootSurfaceInteraction* KSRootSurfaceInteraction::Clone() const
{
    return new KSRootSurfaceInteraction(*this);
}
KSRootSurfaceInteraction::~KSRootSurfaceInteraction() = default;

void KSRootSurfaceInteraction::ExecuteInteraction(const KSParticle& anInitialParticle, KSParticle& aFinalParticle,
                                                  KSParticleQueue& aSecondaries)
{
    if (fSurfaceInteraction == nullptr) {
        intmsg(eError) << "<" << GetName() << "> cannot execute interaction with no surface interaction set" << eom;
    }

    try {
        intmsg_debug("<" << GetName() << "> executing surface interaction <" << fSurfaceInteraction->GetName() << "> at " << anInitialParticle.GetPosition() << eom);
        fSurfaceInteraction->ExecuteInteraction(anInitialParticle, aFinalParticle, aSecondaries);
    }
    catch (KSException const& e) {
        throw KSInteractionError().Nest(e)
            << "Failed to execute surface interaction <" << fSurfaceInteraction->GetName() << ">.";
    }
    return;
}

void KSRootSurfaceInteraction::SetSurfaceInteraction(KSSurfaceInteraction* aSurfaceInteraction)
{
    if (fSurfaceInteraction != nullptr) {
        intmsg(eError) << "<" << GetName() << "> tried to set surface interaction <" << aSurfaceInteraction->GetName()
                       << "> with surface interaction <" << fSurfaceInteraction->GetName() << "> already set" << eom;
        return;
    }
    intmsg_debug("<" << GetName() << "> setting surface interaction <" << aSurfaceInteraction->GetName() << ">" << eom);
    fSurfaceInteraction = aSurfaceInteraction;
    return;
}
void KSRootSurfaceInteraction::ClearSurfaceInteraction(KSSurfaceInteraction* aSurfaceInteraction)
{
    if (fSurfaceInteraction != aSurfaceInteraction) {
        intmsg(eError) << "<" << GetName() << "> tried to remove surface interaction <"
                       << aSurfaceInteraction->GetName() << "> with surface interaction <"
                       << fSurfaceInteraction->GetName() << "> already set" << eom;
        return;
    }
    intmsg_debug("<" << GetName() << "> clearing surface interaction <" << aSurfaceInteraction->GetName() << ">"
                     << eom);
    fSurfaceInteraction = nullptr;
    return;
}

void KSRootSurfaceInteraction::SetStep(KSStep* aStep)
{
    fStep = aStep;
    fTerminatorParticle = &(aStep->InitialParticle());
    fInteractionParticle = &(aStep->InteractionParticle());
    fParticleQueue = &(aStep->ParticleQueue());
    return;
}

void KSRootSurfaceInteraction::ExecuteInteraction()
{
    if (fSurfaceInteraction == nullptr) {
        *fInteractionParticle = *fTerminatorParticle;
        fStep->SurfaceNavigationFlag() = false;
        fStep->SurfaceInteractionName().clear();

        fStep->ContinuousTime() = 0.;
        fStep->ContinuousLength() = 0.;
        fStep->ContinuousEnergyChange() = 0.;
        fStep->ContinuousMomentumChange() = 0.;
        fStep->DiscreteSecondaries() = 0;
        fStep->DiscreteEnergyChange() = 0.;
        fStep->DiscreteMomentumChange() = 0.;

        intmsg_debug("surface interaction:" << eom);
        intmsg_debug("  no surface interactions active" << eom);
        intmsg_debug("  step continuous time: <" << fStep->ContinuousTime() << ">" << eom);
        intmsg_debug("  step continuous length: <" << fStep->ContinuousLength() << ">" << eom);
        intmsg_debug("  step continuous energy change: <" << fStep->ContinuousEnergyChange() << ">" << eom);
        intmsg_debug("  step continuous momentum change: <" << fStep->ContinuousMomentumChange() << ">" << eom);
        intmsg_debug("  step discrete secondaries: <" << fStep->DiscreteSecondaries() << ">" << eom);
        intmsg_debug("  step discrete energy change: <" << fStep->DiscreteEnergyChange() << ">" << eom);
        intmsg_debug("  step discrete momentum change: <" << fStep->DiscreteMomentumChange() << ">" << eom);

        intmsg_debug("surface interaction interaction particle state: " << eom);
        intmsg_debug(
            "  interaction particle space: <"
            << (fInteractionParticle->GetCurrentSpace() ? fInteractionParticle->GetCurrentSpace()->GetName() : "")
            << ">" << eom);
        intmsg_debug(
            "  interaction particle surface: <"
            << (fInteractionParticle->GetCurrentSurface() ? fInteractionParticle->GetCurrentSurface()->GetName() : "")
            << ">" << eom);
        intmsg_debug("  interaction particle time: <" << fInteractionParticle->GetTime() << ">" << eom);
        intmsg_debug("  interaction particle length: <" << fInteractionParticle->GetLength() << ">" << eom);
        intmsg_debug("  interaction particle position: <" << fInteractionParticle->GetPosition().X() << ", "
                                                          << fInteractionParticle->GetPosition().Y() << ", "
                                                          << fInteractionParticle->GetPosition().Z() << ">" << eom);
        intmsg_debug("  interaction particle momentum: <" << fInteractionParticle->GetMomentum().X() << ", "
                                                          << fInteractionParticle->GetMomentum().Y() << ", "
                                                          << fInteractionParticle->GetMomentum().Z() << ">" << eom);
        intmsg_debug("  interaction particle kinetic energy: <" << fInteractionParticle->GetKineticEnergy_eV() << ">"
                                                                << eom);
        intmsg_debug("  interaction particle electric field: <" << fInteractionParticle->GetElectricField().X() << ","
                                                                << fInteractionParticle->GetElectricField().Y() << ","
                                                                << fInteractionParticle->GetElectricField().Z() << ">"
                                                                << eom);
        intmsg_debug("  interaction particle magnetic field: <" << fInteractionParticle->GetMagneticField().X() << ","
                                                                << fInteractionParticle->GetMagneticField().Y() << ","
                                                                << fInteractionParticle->GetMagneticField().Z() << ">"
                                                                << eom);
        intmsg_debug("  interaction particle angle to magnetic field: <" << fInteractionParticle->GetPolarAngleToB()
                                                                         << ">" << eom);
        intmsg_debug("  interaction particle spin: " << fInteractionParticle->GetSpin() << eom);
        intmsg_debug("  interaction particle spin0: <" << fInteractionParticle->GetSpin0() << ">" << eom);
        intmsg_debug("  interaction particle aligned spin: <" << fInteractionParticle->GetAlignedSpin() << ">" << eom);
        intmsg_debug("  interaction particle spin angle: <" << fInteractionParticle->GetSpinAngle() << ">" << eom);

        return;
    }

    ExecuteInteraction(*fTerminatorParticle, *fInteractionParticle, *fParticleQueue);
    fStep->SurfaceNavigationFlag() = true;
    fInteractionParticle->ReleaseLabel(fStep->SurfaceInteractionName());

    fStep->ContinuousTime() = 0.;
    fStep->ContinuousLength() = 0.;
    fStep->ContinuousEnergyChange() = 0.;
    fStep->ContinuousMomentumChange() = 0.;
    fStep->DiscreteSecondaries() = fParticleQueue->size();
    fStep->DiscreteEnergyChange() =
        fInteractionParticle->GetKineticEnergy_eV() - fTerminatorParticle->GetKineticEnergy_eV();
    fStep->DiscreteMomentumChange() =
        (fInteractionParticle->GetMomentum() - fTerminatorParticle->GetMomentum()).Magnitude();

    intmsg_debug("surface interaction:" << eom);
    intmsg_debug("  surface interaction name: <" << fStep->GetSurfaceInteractionName() << ">" << eom);
    intmsg_debug("  step continuous time: <" << fStep->ContinuousTime() << ">" << eom);
    intmsg_debug("  step continuous length: <" << fStep->ContinuousLength() << ">" << eom);
    intmsg_debug("  step continuous energy change: <" << fStep->ContinuousEnergyChange() << ">" << eom);
    intmsg_debug("  step continuous momentum change: <" << fStep->ContinuousMomentumChange() << ">" << eom);
    intmsg_debug("  step discrete secondaries: <" << fStep->DiscreteSecondaries() << ">" << eom);
    intmsg_debug("  step discrete energy change: <" << fStep->DiscreteEnergyChange() << ">" << eom);
    intmsg_debug("  step discrete momentum change: <" << fStep->DiscreteMomentumChange() << ">" << eom);

    intmsg_debug("surface interaction reflection interaction particle state: " << eom);
    intmsg_debug("  interaction particle space: <"
                 << (fInteractionParticle->GetCurrentSpace() ? fInteractionParticle->GetCurrentSpace()->GetName() : "")
                 << ">" << eom);
    intmsg_debug(
        "  interaction particle surface: <"
        << (fInteractionParticle->GetCurrentSurface() ? fInteractionParticle->GetCurrentSurface()->GetName() : "")
        << ">" << eom);
    intmsg_debug("  interaction particle time: <" << fInteractionParticle->GetTime() << ">" << eom);
    intmsg_debug("  interaction particle length: <" << fInteractionParticle->GetLength() << ">" << eom);
    intmsg_debug("  interaction particle position: <" << fInteractionParticle->GetPosition().X() << ", "
                                                      << fInteractionParticle->GetPosition().Y() << ", "
                                                      << fInteractionParticle->GetPosition().Z() << ">" << eom);
    intmsg_debug("  interaction particle momentum: <" << fInteractionParticle->GetMomentum().X() << ", "
                                                      << fInteractionParticle->GetMomentum().Y() << ", "
                                                      << fInteractionParticle->GetMomentum().Z() << ">" << eom);
    intmsg_debug("  interaction particle kinetic energy: <" << fInteractionParticle->GetKineticEnergy_eV() << ">"
                                                            << eom);
    intmsg_debug("  interaction particle electric field: <"
                 << fInteractionParticle->GetElectricField().X() << "," << fInteractionParticle->GetElectricField().Y()
                 << "," << fInteractionParticle->GetElectricField().Z() << ">" << eom);
    intmsg_debug("  interaction particle magnetic field: <"
                 << fInteractionParticle->GetMagneticField().X() << "," << fInteractionParticle->GetMagneticField().Y()
                 << "," << fInteractionParticle->GetMagneticField().Z() << ">" << eom);
    intmsg_debug("  interaction particle angle to magnetic field: <" << fInteractionParticle->GetPolarAngleToB() << ">"
                                                                     << eom);
    intmsg_debug("  interaction particle spin: " << fInteractionParticle->GetSpin() << eom);
    intmsg_debug("  interaction particle spin0: <" << fInteractionParticle->GetSpin0() << ">" << eom);
    intmsg_debug("  interaction particle aligned spin: <" << fInteractionParticle->GetAlignedSpin() << ">" << eom);
    intmsg_debug("  interaction particle spin angle: <" << fInteractionParticle->GetSpinAngle() << ">" << eom);

    return;
}

STATICINT sKSRootSurfaceInteractionDict = KSDictionary<KSRootSurfaceInteraction>::AddCommand(
    &KSRootSurfaceInteraction::SetSurfaceInteraction, &KSRootSurfaceInteraction::ClearSurfaceInteraction,
    "set_surface_interaction", "clear_surface_interaction");

}  // namespace Kassiopeia
