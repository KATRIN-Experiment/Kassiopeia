#include "KSIntDecayCalculator.h"

namespace Kassiopeia
{


KSIntDecayCalculator::KSIntDecayCalculator() : fStepNDecays(0), fStepEnergyLoss(0.0) {}

KSIntDecayCalculator::~KSIntDecayCalculator() = default;

void KSIntDecayCalculator::PullDeupdateComponent()
{
    fStepNDecays = 0;
    fStepEnergyLoss = 0.0;
}

void KSIntDecayCalculator::PushDeupdateComponent()
{
    fStepNDecays = 0;
    fStepEnergyLoss = 0.0;
}

STATICINT sKSIntDecayCalculatorDict =
    KSDictionary<KSIntDecayCalculator>::AddComponent(&KSIntDecayCalculator::GetStepNDecays, "step_number_of_decays") +

    KSDictionary<KSIntDecayCalculator>::AddComponent(&KSIntDecayCalculator::GetStepEnergyLoss, "step_energy_loss");

} /* namespace Kassiopeia */
