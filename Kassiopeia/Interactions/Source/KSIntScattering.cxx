#include "KSIntScattering.h"

#include "KRandom.h"
#include "KSIntCalculator.h"
#include "KSIntDensity.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

#include <limits>
using std::numeric_limits;

using KGeoBag::KThreeVector;

namespace Kassiopeia
{

KSIntScattering::KSIntScattering() :
    fSplit(false),
    fDensity(nullptr),
    fCalculator(nullptr),
    fCalculators(),
    fCrossSections(),
    fEnhancement(1.)
{}
KSIntScattering::KSIntScattering(const KSIntScattering& aCopy) :
    KSComponent(aCopy),
    KSComponentTemplate<KSIntScattering, KSSpaceInteraction>(aCopy),
    fSplit(aCopy.fSplit),
    fDensity(aCopy.fDensity),
    fCalculator(aCopy.fCalculator),
    fCalculators(aCopy.fCalculators),
    fCrossSections(aCopy.fCrossSections),
    fEnhancement(aCopy.fEnhancement)
{}
KSIntScattering* KSIntScattering::Clone() const
{
    return new KSIntScattering(*this);
}
KSIntScattering::~KSIntScattering() = default;
//{
//        for( unsigned int tIndex = 0; tIndex < fCalculators.size(); tIndex++ )
//        {
//            delete (fCalculators.at( tIndex ));
//        }
//        fCalculators.clear();
//}


void KSIntScattering::CalculateAverageCrossSection(const KSParticle& aTrajectoryInitialParticle,
                                                   const KSParticle& aTrajectoryFinalParticle,
                                                   double& anAverageCrossSection)
{
    double tInitialCrossSection;
    double tFinalCrossSection;
    anAverageCrossSection = 0;
    for (unsigned int tIndex = 0; tIndex < fCalculators.size(); tIndex++) {
        fCalculators.at(tIndex)->CalculateCrossSection(aTrajectoryInitialParticle, tInitialCrossSection);
        fCalculators.at(tIndex)->CalculateCrossSection(aTrajectoryFinalParticle, tFinalCrossSection);
        fCrossSections.at(tIndex) = 0.5 * (tInitialCrossSection + tFinalCrossSection);
        anAverageCrossSection += fCrossSections.at(tIndex);
    }
}


void KSIntScattering::DiceCalculator(const double& anAverageCrossSection)
{
    double tProbability = KRandom::GetInstance().Uniform(0., anAverageCrossSection);
    for (unsigned int tIndex = 0; tIndex < fCalculators.size(); tIndex++) {
        if (fCrossSections.at(tIndex) > 0) {
            tProbability -= fCrossSections.at(tIndex);
            if (tProbability < 0.) {
                fCalculator = fCalculators.at(tIndex);
                break;
            }
        }
        else {
            continue;
        }
    }
}

void KSIntScattering::CalculateInteraction(const KSTrajectory& aTrajectory,
                                           const KSParticle& aTrajectoryInitialParticle,
                                           const KSParticle& aTrajectoryFinalParticle,
                                           const KThreeVector& /*aTrajectoryCenter*/,
                                           const double& /*aTrajectoryRadius*/, const double& aTrajectoryTimeStep,
                                           KSParticle& anInteractionParticle, double& aTimeStep, bool& aFlag)
{
    intmsg_debug("scattering interaction <" << this->GetName() << "> calculating interaction:" << eom);

    double tInitialSpeed = aTrajectoryInitialParticle.GetSpeed();
    double tFinalSpeed = aTrajectoryFinalParticle.GetSpeed();
    double tAverageSpeed = .5 * (tInitialSpeed + tFinalSpeed);

    intmsg_debug("  average speed: <" << tAverageSpeed << ">" << eom);

    double tInitialDensity = 0.;
    fDensity->CalculateDensity(aTrajectoryInitialParticle, tInitialDensity);
    double tFinalDensity = 0.;
    fDensity->CalculateDensity(aTrajectoryFinalParticle, tFinalDensity);
    double tAverageDensity = .5 * (tInitialDensity + tFinalDensity);

    intmsg_debug("  average density: <" << tAverageDensity << ">" << eom);


    if (tAverageDensity <= 0.0) {
        aTimeStep = aTrajectoryTimeStep;
        aFlag = false;
        return;
    }

    double tAverageTotalCrossSection = 0.;
    CalculateAverageCrossSection(aTrajectoryInitialParticle, aTrajectoryFinalParticle, tAverageTotalCrossSection);

    intmsg_debug("  average cross section: <" << tAverageTotalCrossSection << ">" << eom);

    double tTime;
    double tProbability;
    double tDenominator = tAverageDensity * tAverageTotalCrossSection * tAverageSpeed * fEnhancement;
    if (tDenominator > 0.) {
        tProbability = KRandom::GetInstance().Uniform(0., 1.);
        tTime = -1. * log(1. - tProbability) / tDenominator;
    }
    else {
        tTime = numeric_limits<double>::max();
    }

    intmsg_debug("  scattering time: <" << tTime << ">" << eom);

    if (tTime > aTrajectoryTimeStep) {
        fCalculator = nullptr;

        anInteractionParticle = aTrajectoryFinalParticle;
        aTimeStep = aTrajectoryTimeStep;
        aFlag = false;

        intmsg_debug("  no scattering process occurred" << eom);
    }
    else {
        DiceCalculator(tAverageTotalCrossSection);

        anInteractionParticle = aTrajectoryInitialParticle;
        aTrajectory.ExecuteTrajectory(tTime, anInteractionParticle);
        aTimeStep = tTime;
        aFlag = true;

        intmsg_debug("  scattering process <" << fCalculator->GetName() << "> may occur" << eom);
    }

    return;
}

void KSIntScattering::ExecuteInteraction(const KSParticle& anInteractionParticle, KSParticle& aFinalParticle,
                                         KSParticleQueue& aSecondaries) const
{
    if (fCalculator != nullptr) {
        if (fSplit == true) {
            auto* tSplitParticle = new KSParticle();
            *tSplitParticle = aFinalParticle;

            fCalculator->ExecuteInteraction(anInteractionParticle, *tSplitParticle, aSecondaries);
            aSecondaries.push_back(tSplitParticle);

            aFinalParticle.SetActive(false);
            aFinalParticle.SetLabel(GetName());
        }
        else {
            fCalculator->ExecuteInteraction(anInteractionParticle, aFinalParticle, aSecondaries);
        }
    }
    else {
        aFinalParticle = anInteractionParticle;
    }
    return;
}

void KSIntScattering::SetSplit(const bool& aSplit)
{
    fSplit = aSplit;
    return;
}
const bool& KSIntScattering::GetSplit() const
{
    return fSplit;
}

void KSIntScattering::SetDensity(KSIntDensity* aDensityCalculator)
{
    if (fDensity == nullptr) {
        fDensity = aDensityCalculator;
        return;
    }
    intmsg(eError) << "cannot set density calculator <" << aDensityCalculator->GetName() << "> to scattering module <"
                   << GetName() << ">" << eom;
    return;
}
void KSIntScattering::ClearDensity(KSIntDensity* aDensityCalculator)
{
    if (fDensity == aDensityCalculator) {
        fDensity = nullptr;
        return;
    }
    intmsg(eError) << "cannot clear density calculator <" << aDensityCalculator->GetName()
                   << "> from scattering module <" << GetName() << ">" << eom;
    return;
}

void KSIntScattering::AddCalculator(KSIntCalculator* aScatteringCalculator)
{
    KSIntCalculator* tCalculator;
    std::vector<KSIntCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator == aScatteringCalculator) {
            intmsg(eError) << "could not add scattering calculator <" << aScatteringCalculator->GetName()
                           << "> to scattering module <" << GetName() << ">" << eom;
            return;
        }
    }
    fCalculators.push_back(aScatteringCalculator);
    fCrossSections.resize(fCalculators.size(), 0.);
    intmsg_debug("added scattering calculator <" << aScatteringCalculator->GetName() << "> to scattering module <"
                                                 << GetName() << ">" << eom);
    return;
}
void KSIntScattering::RemoveCalculator(KSIntCalculator* aScatteringCalculator)
{
    KSIntCalculator* tCalculator;
    std::vector<KSIntCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator == aScatteringCalculator) {
            fCalculators.erase(tIt);
            fCrossSections.resize(fCalculators.size(), 0.);
            intmsg_debug("removed scattering calculator <" << aScatteringCalculator->GetName()
                                                           << "> to scattering module <" << GetName() << ">" << eom);
            return;
        }
    }
    intmsg(eError) << "could not remove scattering calculator <" << aScatteringCalculator->GetName()
                   << "> to scattering module <" << GetName() << ">" << eom;
    return;
}

void KSIntScattering::SetEnhancement(double anEnhancement)
{
    fEnhancement = anEnhancement;
}

void KSIntScattering::InitializeComponent()
{
    KSIntCalculator* tCalculator;
    std::vector<KSIntCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->Initialize();
        }
    }
    if (fDensity != nullptr) {
        fDensity->Initialize();
    }
    return;
}

void KSIntScattering::DeinitializeComponent()
{
    KSIntCalculator* tCalculator;
    std::vector<KSIntCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->Deinitialize();
        }
    }
    if (fDensity != nullptr) {
        fDensity->Deinitialize();
    }
    return;
}

void KSIntScattering::ActivateComponent()
{
    KSIntCalculator* tCalculator;
    std::vector<KSIntCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->Activate();
        }
    }
    if (fDensity != nullptr) {
        fDensity->Activate();
    }
    return;
}

void KSIntScattering::DeactivateComponent()
{
    KSIntCalculator* tCalculator;
    std::vector<KSIntCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->Deactivate();
        }
    }
    if (fDensity != nullptr) {
        fDensity->Deactivate();
    }
    return;
}

void KSIntScattering::PushUpdateComponent()
{
    KSIntCalculator* tCalculator;
    std::vector<KSIntCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->PushUpdate();
        }
    }
    if (fDensity != nullptr) {
        fDensity->PushUpdate();
    }
    return;
}

void KSIntScattering::PushDeupdateComponent()
{
    KSIntCalculator* tCalculator;
    std::vector<KSIntCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->PushDeupdate();
        }
    }
    if (fDensity != nullptr) {
        fDensity->PushDeupdate();
    }
    return;
}

STATICINT sKSIntScatteringDict =
    KSDictionary<KSIntScattering>::AddCommand(&KSIntScattering::AddCalculator, &KSIntScattering::RemoveCalculator,
                                              "add_calculator", "remove_calculator") +
    KSDictionary<KSIntScattering>::AddCommand(&KSIntScattering::SetDensity, &KSIntScattering::ClearDensity,
                                              "set_density", "clear_density");

}  // namespace Kassiopeia
