#include "KSIntDecay.h"

#include "KRandom.h"
#include "KSInteractionsMessage.h"
using katrin::KRandom;

//#include <bits/stl_algo.h>

#include <algorithm>
using std::numeric_limits;

using katrin::KThreeVector;

namespace Kassiopeia
{

KSIntDecay::KSIntDecay() : fSplit(false), fCalculator(nullptr), fCalculators(), fLifeTimes(), fEnhancement(1.) {}

KSIntDecay::KSIntDecay(const KSIntDecay& aCopy) :
    KSComponent(aCopy),
    KSComponentTemplate<KSIntDecay, KSSpaceInteraction>(aCopy),
    fSplit(aCopy.fSplit),
    fCalculator(aCopy.fCalculator),
    fCalculators(aCopy.fCalculators),
    fLifeTimes(aCopy.fLifeTimes),
    fEnhancement(aCopy.fEnhancement)
{}

KSIntDecay* KSIntDecay::Clone() const
{
    return new KSIntDecay(*this);
}

KSIntDecay::~KSIntDecay() = default;
//{
//        for( unsigned int tIndex = 0; tIndex < fCalculators.size(); tIndex++ )
//        {
//            delete (fCalculators.at( tIndex ));
//        }
//        fCalculators.clear();
//}

std::vector<double> KSIntDecay::CalculateLifetimes(const KSParticle& aTrajectoryInitialParticle)
{
    double tInitialLifeTime;
    std::vector<double> tLifetimes(fCalculators.size(), std::numeric_limits<double>::max());

    for (unsigned int tIndex = 0; tIndex < fCalculators.size(); tIndex++) {
        fCalculators.at(tIndex)->CalculateLifeTime(aTrajectoryInitialParticle, tInitialLifeTime);
        intmsg(eDebug) << fCalculators.at(tIndex)->GetName() << " initial life time " << tInitialLifeTime << eom;

        tLifetimes.at(tIndex) = tInitialLifeTime;
    }
    return tLifetimes;
}


void KSIntDecay::CalculateInteraction(const KSTrajectory& aTrajectory, const KSParticle& aTrajectoryInitialParticle,
                                      const KSParticle& aTrajectoryFinalParticle,
                                      const KThreeVector& /*aTrajectoryCenter*/,
                                      const double& /*aTrajectoryRadius*/, const double& aTrajectoryTimeStep,
                                      KSParticle& anInteractionParticle, double& aTimeStep, bool& aFlag)
{
    intmsg_debug("decay interaction <" << this->GetName() << "> calculating interaction:" << eom);

    std::vector<double> tLifetimes = CalculateLifetimes(aTrajectoryInitialParticle);
    std::vector<double> tTimes(fCalculators.size(), numeric_limits<double>::max());
    for (unsigned int tIndex = 0; tIndex < fCalculators.size(); tIndex++) {
        tTimes.at(tIndex) = -1. * log(KRandom::GetInstance().Uniform(0., 1.)) * tLifetimes.at(tIndex) / fEnhancement;
        intmsg_debug("  decay time " << fCalculators.at(tIndex)->GetName() << ": " << tTimes.at(tIndex) << eom);
    }

    unsigned long tSmallest = (unsigned long) (std::min_element(tTimes.begin(), tTimes.end()) - tTimes.begin());
    intmsg_debug("  samllest decay time: <" << tTimes.at(tSmallest) << ">" << eom);
    intmsg_debug("  aTrajectoryTimeStep: <" << aTrajectoryTimeStep << ">" << eom);

    if (tTimes.at(tSmallest) > aTrajectoryTimeStep) {
        fCalculator = nullptr;

        anInteractionParticle = aTrajectoryFinalParticle;
        aTimeStep = aTrajectoryTimeStep;
        aFlag = false;

        intmsg_debug("  no decay process occurred" << eom);
    }
    else {
        fCalculator = fCalculators.at(tSmallest);

        anInteractionParticle = aTrajectoryInitialParticle;
        aTrajectory.ExecuteTrajectory(tTimes.at(tSmallest), anInteractionParticle);
        aTimeStep = tTimes.at(tSmallest);
        aFlag = true;

        intmsg_debug("  decay process <" << fCalculator->GetName() << "> may occur" << eom);
    }

    return;
}

void KSIntDecay::ExecuteInteraction(const KSParticle& anInteractionParticle, KSParticle& aFinalParticle,
                                    KSParticleQueue& aSecondaries) const
{
    if (fCalculator != nullptr) {
        if (fSplit) {
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

void KSIntDecay::SetSplit(const bool& aSplit)
{
    fSplit = aSplit;
    return;
}

const bool& KSIntDecay::GetSplit() const
{
    return fSplit;
}

void KSIntDecay::AddCalculator(KSIntDecayCalculator* aScatteringCalculator)
{
    KSIntDecayCalculator* tCalculator;
    std::vector<KSIntDecayCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator == aScatteringCalculator) {
            intmsg(eError) << "could not add scattering calculator <" << aScatteringCalculator->GetName()
                           << "> to scattering module <" << GetName() << ">" << eom;
            return;
        }
    }
    fCalculators.push_back(aScatteringCalculator);
    fLifeTimes.resize(fCalculators.size(), 0.);
    intmsg_debug("added scattering calculator <" << aScatteringCalculator->GetName() << "> to scattering module <"
                                                 << GetName() << ">" << eom);
    return;
}

void KSIntDecay::RemoveCalculator(KSIntDecayCalculator* aScatteringCalculator)
{
    KSIntDecayCalculator* tCalculator;
    std::vector<KSIntDecayCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator == aScatteringCalculator) {
            fCalculators.erase(tIt);
            fLifeTimes.resize(fCalculators.size(), 0.);
            intmsg_debug("removed scattering calculator <" << aScatteringCalculator->GetName()
                                                           << "> to scattering module <" << GetName() << ">" << eom);
            return;
        }
    }
    intmsg(eError) << "could not remove scattering calculator <" << aScatteringCalculator->GetName()
                   << "> to scattering module <" << GetName() << ">" << eom;
    return;
}

void KSIntDecay::SetEnhancement(double anEnhancement)
{
    fEnhancement = anEnhancement;
}

void KSIntDecay::InitializeComponent()
{
    KSIntDecayCalculator* tCalculator;
    std::vector<KSIntDecayCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->Initialize();
        }
    }
    return;
}

void KSIntDecay::DeinitializeComponent()
{
    KSIntDecayCalculator* tCalculator;
    std::vector<KSIntDecayCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->Deinitialize();
        }
    }
    return;
}

void KSIntDecay::ActivateComponent()
{
    KSIntDecayCalculator* tCalculator;
    std::vector<KSIntDecayCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->Activate();
        }
    }
    return;
}

void KSIntDecay::DeactivateComponent()
{
    KSIntDecayCalculator* tCalculator;
    std::vector<KSIntDecayCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->Deactivate();
        }
    }
    return;
}

void KSIntDecay::PushUpdateComponent()
{
    KSIntDecayCalculator* tCalculator;
    std::vector<KSIntDecayCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->PushUpdate();
        }
    }
    return;
}

void KSIntDecay::PushDeupdateComponent()
{
    KSIntDecayCalculator* tCalculator;
    std::vector<KSIntDecayCalculator*>::iterator tIt;
    for (tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        tCalculator = (*tIt);
        if (tCalculator != nullptr) {
            tCalculator->PushDeupdate();
        }
    }
    return;
}

STATICINT sKSIntDecayDict = KSDictionary<KSIntDecay>::AddCommand(
    &KSIntDecay::AddCalculator, &KSIntDecay::RemoveCalculator, "add_calculator", "remove_calculator");

}  // namespace Kassiopeia
