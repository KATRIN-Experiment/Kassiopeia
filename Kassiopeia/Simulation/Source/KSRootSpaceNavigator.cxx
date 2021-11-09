#include "KSRootSpaceNavigator.h"

#include "KSException.h"
#include "KSNavigatorsMessage.h"

#include <limits>

namespace Kassiopeia
{

KSRootSpaceNavigator::KSRootSpaceNavigator() :
    fSpaceNavigator(nullptr),
    fStep(nullptr),
    fTerminatorParticle(nullptr),
    fTrajectoryParticle(nullptr),
    fNavigationParticle(nullptr),
    fFinalParticle(nullptr),
    fParticleQueue(nullptr),
    fTrajectory(nullptr)
{}
KSRootSpaceNavigator::KSRootSpaceNavigator(const KSRootSpaceNavigator& aCopy) :
    KSComponent(aCopy),
    fSpaceNavigator(aCopy.fSpaceNavigator),
    fStep(aCopy.fStep),
    fTerminatorParticle(aCopy.fTerminatorParticle),
    fTrajectoryParticle(aCopy.fTrajectoryParticle),
    fNavigationParticle(aCopy.fNavigationParticle),
    fFinalParticle(aCopy.fFinalParticle),
    fParticleQueue(aCopy.fParticleQueue),
    fTrajectory(aCopy.fTrajectory)
{}
KSRootSpaceNavigator* KSRootSpaceNavigator::Clone() const
{
    return new KSRootSpaceNavigator(*this);
}
KSRootSpaceNavigator::~KSRootSpaceNavigator() = default;

void KSRootSpaceNavigator::CalculateNavigation(const KSTrajectory& aTrajectory,
                                               const KSParticle& aTrajectoryInitialParticle,
                                               const KSParticle& aTrajectoryFinalParticle,
                                               const KGeoBag::KThreeVector& aTrajectoryCenter,
                                               const double& aTrajectoryRadius,
                                               const double& aTrajectoryStep,
                                               KSParticle& aNavigationParticle,
                                               double& aNavigationStep,
                                               bool& aNavigationFlag)
{
    if (fSpaceNavigator == nullptr) {
        navmsg(eError) << "<" << GetName() << "> cannot calculate navigation with no space navigator set" << eom;
    }

    try {
        navmsg_debug("<" << GetName() << "> calculating space navigator <" << fSpaceNavigator->GetName() << "> at " << aTrajectoryCenter << eom);
        fSpaceNavigator->CalculateNavigation(aTrajectory,
                                             aTrajectoryInitialParticle,
                                             aTrajectoryFinalParticle,
                                             aTrajectoryCenter,
                                             aTrajectoryRadius,
                                             aTrajectoryStep,
                                             aNavigationParticle,
                                             aNavigationStep,
                                             aNavigationFlag);
    }
    catch (KSException const& e) {
        throw KSNavigatorError().Nest(e) << "Space navigator calculation error";
    }
    return;
}
void KSRootSpaceNavigator::ExecuteNavigation(const KSParticle& aNavigationParticle,
                                             KSParticle& aFinalParticle,
                                             KSParticleQueue& aSecondaries) const
{
    if (fSpaceNavigator == nullptr) {
        navmsg(eError) << "<" << GetName() << "> cannot execute navigation with no space navigator set" << eom;
    }

    try {
        navmsg_debug("<" << GetName() << "> executing space navigator <" << fSpaceNavigator->GetName() << "> at " << aNavigationParticle.GetPosition() << eom);
        fSpaceNavigator->ExecuteNavigation(aNavigationParticle, aFinalParticle, aSecondaries);
    }
    catch (KSException const& e) {
        throw KSNavigatorError().Nest(e) << "Failed to execute space navigator <" << fSpaceNavigator->GetName() << ">.";
    }
    return;
}
void KSRootSpaceNavigator::FinalizeNavigation(KSParticle& aFinalParticle) const
{
    if (fSpaceNavigator == nullptr) {
        navmsg(eError) << "<" << GetName() << "> cannot finalize navigation with no space navigator set" << eom;
    }

    try {
        fSpaceNavigator->FinalizeNavigation(aFinalParticle);
    }
    catch (KSException const& e) {
        throw KSNavigatorError().Nest(e) << "Failed to finalize space navigator <" << fSpaceNavigator->GetName()
                                         << ">.";
    }
    return;
}
void KSRootSpaceNavigator::StartNavigation(KSParticle& aParticle, KSSpace* aRoot)
{
    if (fSpaceNavigator == nullptr) {
        navmsg(eError) << "<" << GetName() << "> cannot start navigation with no space navigator set" << eom;
    }

    try {
        fSpaceNavigator->StartNavigation(aParticle, aRoot);
    }
    catch (KSException const& e) {
        throw KSNavigatorError().Nest(e) << "Failed to start space navigation in <" << aRoot->GetName() << ">.";
    }
    return;
}
void KSRootSpaceNavigator::StopNavigation(KSParticle& aParticle, KSSpace* aRoot)
{
    if (fSpaceNavigator == nullptr) {
        navmsg(eError) << "<" << GetName() << "> cannot stop navigation with no space navigator set" << eom;
    }

    try {
        fSpaceNavigator->StopNavigation(aParticle, aRoot);
    }
    catch (KSException const& e) {
        throw KSNavigatorError().Nest(e) << "Failed to stop space navigation in <" << aRoot->GetName() << ">.";
    }
    return;
}

void KSRootSpaceNavigator::SetSpaceNavigator(KSSpaceNavigator* aSpaceNavigator)
{
    if (fSpaceNavigator != nullptr) {
        navmsg(eError) << "<" << GetName() << "> tried to set space navigator <" << aSpaceNavigator->GetName()
                       << "> with space navigator <" << fSpaceNavigator->GetName() << "> already set" << eom;
        return;
    }
    navmsg_debug("<" << GetName() << "> setting space navigator <" << aSpaceNavigator->GetName() << ">" << eom);
    fSpaceNavigator = aSpaceNavigator;
    return;
}
void KSRootSpaceNavigator::ClearSpaceNavigator(KSSpaceNavigator* aSpaceNavigator)
{
    if (fSpaceNavigator != aSpaceNavigator) {
        navmsg(eError) << "<" << GetName() << "> tried to remove space navigator <" << aSpaceNavigator->GetName()
                       << "> with space navigator <" << fSpaceNavigator->GetName() << "> already set" << eom;
        return;
    }
    navmsg_debug("<" << GetName() << "> clearing space navigator <" << aSpaceNavigator->GetName() << ">" << eom);
    fSpaceNavigator = nullptr;
    return;
}

void KSRootSpaceNavigator::SetStep(KSStep* aStep)
{
    fStep = aStep;
    fTerminatorParticle = &(aStep->TerminatorParticle());
    fTrajectoryParticle = &(aStep->TrajectoryParticle());
    fNavigationParticle = &(aStep->NavigationParticle());
    fFinalParticle = &(aStep->FinalParticle());
    fParticleQueue = &(aStep->ParticleQueue());
    return;
}
void KSRootSpaceNavigator::SetTrajectory(KSTrajectory* aTrajectory)
{
    fTrajectory = aTrajectory;
    return;
}

void KSRootSpaceNavigator::CalculateNavigation()
{
    *fNavigationParticle = *fTrajectoryParticle;

    CalculateNavigation(*fTrajectory,
                        *fTerminatorParticle,
                        *fTrajectoryParticle,
                        fStep->TrajectoryCenter(),
                        fStep->TrajectoryRadius(),
                        fStep->TrajectoryStep(),
                        *fNavigationParticle,
                        fStep->SpaceNavigationStep(),
                        fStep->SpaceNavigationFlag());

    if (! fNavigationParticle->IsValid())
        throw KSNavigatorError() << "invalid particle state after navigation calculation";

    if (fStep->SpaceNavigationFlag() == true) {
        navmsg_debug("space navigation calculation:" << eom);
        navmsg_debug("  space navigation may occur" << eom);
    }
    else {
        navmsg_debug("space navigation calculation:" << eom);
        navmsg_debug("  space navigation will not occur" << eom);
    }

    navmsg_debug("space navigation calculation particle state: " << eom);
    navmsg_debug("  final particle space: <"
                 << (fNavigationParticle->GetCurrentSpace() ? fNavigationParticle->GetCurrentSpace()->GetName() : "")
                 << ">" << eom);
    navmsg_debug("  final particle surface: <"
                 << (fNavigationParticle->GetCurrentSurface() ? fNavigationParticle->GetCurrentSurface()->GetName()
                                                              : "")
                 << ">" << eom);
    navmsg_debug("  final particle side: <"
                 << (fNavigationParticle->GetCurrentSide() ? fNavigationParticle->GetCurrentSide()->GetName() : "")
                 << ">" << eom);
    navmsg_debug("  final particle time: <" << fNavigationParticle->GetTime() << ">" << eom);
    navmsg_debug("  final particle length: <" << fNavigationParticle->GetLength() << ">" << eom);
    navmsg_debug("  final particle position: <" << fNavigationParticle->GetPosition().X() << ", "
                                                << fNavigationParticle->GetPosition().Y() << ", "
                                                << fNavigationParticle->GetPosition().Z() << ">" << eom);
    navmsg_debug("  final particle momentum: <" << fNavigationParticle->GetMomentum().X() << ", "
                                                << fNavigationParticle->GetMomentum().Y() << ", "
                                                << fNavigationParticle->GetMomentum().Z() << ">" << eom);
    navmsg_debug("  final particle kinetic energy: <" << fNavigationParticle->GetKineticEnergy_eV() << ">" << eom);
    navmsg_debug("  final particle electric field: <" << fNavigationParticle->GetElectricField().X() << ","
                                                      << fNavigationParticle->GetElectricField().Y() << ","
                                                      << fNavigationParticle->GetElectricField().Z() << ">" << eom);
    navmsg_debug("  final particle magnetic field: <" << fNavigationParticle->GetMagneticField().X() << ","
                                                      << fNavigationParticle->GetMagneticField().Y() << ","
                                                      << fNavigationParticle->GetMagneticField().Z() << ">" << eom);
    navmsg_debug("  final particle angle to magnetic field: <" << fNavigationParticle->GetPolarAngleToB() << ">"
                                                               << eom);
    navmsg_debug("  final particle spin: " << fNavigationParticle->GetSpin() << eom);
    navmsg_debug("  final particle spin0: <" << fNavigationParticle->GetSpin0() << ">" << eom);
    navmsg_debug("  final particle aligned spin: <" << fNavigationParticle->GetAlignedSpin() << ">" << eom);
    navmsg_debug("  final particle spin angle: <" << fNavigationParticle->GetSpinAngle() << ">" << eom);

    return;
}

void KSRootSpaceNavigator::ExecuteNavigation()
{
    ExecuteNavigation(*fNavigationParticle, *fFinalParticle, *fParticleQueue);
    fFinalParticle->ReleaseLabel(fStep->SpaceNavigationName());

    if (! fFinalParticle->IsValid())
        throw KSNavigatorError() << "invalid particle state after navigation execution";

    fStep->ContinuousTime() = fNavigationParticle->GetTime() - fTerminatorParticle->GetTime();
    fStep->ContinuousLength() = fNavigationParticle->GetLength() - fTerminatorParticle->GetLength();
    fStep->ContinuousEnergyChange() =
        fNavigationParticle->GetKineticEnergy_eV() - fTerminatorParticle->GetKineticEnergy_eV();
    fStep->ContinuousMomentumChange() =
        (fNavigationParticle->GetMomentum() - fTerminatorParticle->GetMomentum()).Magnitude();
    fStep->DiscreteSecondaries() = 0;
    fStep->DiscreteEnergyChange() = 0.;
    fStep->DiscreteMomentumChange() = 0.;

    navmsg_debug("space navigation execution:" << eom);
    navmsg_debug("  space navigation name: <" << fStep->SpaceNavigationName() << ">" << eom);
    navmsg_debug("  step continuous time: <" << fStep->ContinuousTime() << ">" << eom);
    navmsg_debug("  step continuous length: <" << fStep->ContinuousLength() << ">" << eom);
    navmsg_debug("  step continuous energy change: <" << fStep->ContinuousEnergyChange() << ">" << eom);
    navmsg_debug("  step continuous momentum change: <" << fStep->ContinuousMomentumChange() << ">" << eom);
    navmsg_debug("  step discrete secondaries: <" << fStep->DiscreteSecondaries() << ">" << eom);
    navmsg_debug("  step discrete energy change: <" << fStep->DiscreteEnergyChange() << ">" << eom);
    navmsg_debug("  step discrete momentum change: <" << fStep->DiscreteMomentumChange() << ">" << eom);

    navmsg_debug("space navigation final particle state: " << eom);
    navmsg_debug("  final particle space: <"
                 << (fFinalParticle->GetCurrentSpace() ? fFinalParticle->GetCurrentSpace()->GetName() : "") << ">"
                 << eom);
    navmsg_debug("  final particle surface: <"
                 << (fFinalParticle->GetCurrentSurface() ? fFinalParticle->GetCurrentSurface()->GetName() : "") << ">"
                 << eom);
    navmsg_debug("  final particle side: <"
                 << (fFinalParticle->GetCurrentSide() ? fFinalParticle->GetCurrentSide()->GetName() : "") << ">"
                 << eom);
    navmsg_debug("  final particle time: <" << fFinalParticle->GetTime() << ">" << eom);
    navmsg_debug("  final particle length: <" << fFinalParticle->GetLength() << ">" << eom);
    navmsg_debug("  final particle position: <" << fFinalParticle->GetPosition().X() << ", "
                                                << fFinalParticle->GetPosition().Y() << ", "
                                                << fFinalParticle->GetPosition().Z() << ">" << eom);
    navmsg_debug("  final particle momentum: <" << fFinalParticle->GetMomentum().X() << ", "
                                                << fFinalParticle->GetMomentum().Y() << ", "
                                                << fFinalParticle->GetMomentum().Z() << ">" << eom);
    navmsg_debug("  final particle kinetic energy: <" << fFinalParticle->GetKineticEnergy_eV() << ">" << eom);
    navmsg_debug("  final particle electric field: <" << fFinalParticle->GetElectricField().X() << ","
                                                      << fFinalParticle->GetElectricField().Y() << ","
                                                      << fFinalParticle->GetElectricField().Z() << ">" << eom);
    navmsg_debug("  final particle magnetic field: <" << fFinalParticle->GetMagneticField().X() << ","
                                                      << fFinalParticle->GetMagneticField().Y() << ","
                                                      << fFinalParticle->GetMagneticField().Z() << ">" << eom);
    navmsg_debug("  final particle angle to magnetic field: <" << fFinalParticle->GetPolarAngleToB() << ">" << eom);
    navmsg_debug("  final particle spin: " << fFinalParticle->GetSpin() << eom);
    navmsg_debug("  final particle spin0: <" << fFinalParticle->GetSpin0() << ">" << eom);
    navmsg_debug("  final particle aligned spin: <" << fFinalParticle->GetAlignedSpin() << ">" << eom);
    navmsg_debug("  final particle spin angle: <" << fFinalParticle->GetSpinAngle() << ">" << eom);

    return;
}

void KSRootSpaceNavigator::FinalizeNavigation()
{
    FinalizeNavigation(*fFinalParticle);
    return;
}

STATICINT sKSRootSpaceNavigatorDict = KSDictionary<KSRootSpaceNavigator>::AddCommand(
    &KSRootSpaceNavigator::SetSpaceNavigator, &KSRootSpaceNavigator::ClearSpaceNavigator, "set_space_navigator",
    "clear_space_navigator");

}  // namespace Kassiopeia
