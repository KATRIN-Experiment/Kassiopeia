#include "KSIntCalculator.h"

namespace Kassiopeia
{


    KSIntCalculator::KSIntCalculator() :
    		fStepNInteractions( 0 ),
    		fStepEnergyLoss( 0.0 )
    {
    }

    KSIntCalculator::~KSIntCalculator()
    {
    }

    void KSIntCalculator::PullDeupdateComponent()
    {
    	fStepNInteractions = 0;
    	fStepEnergyLoss = 0.0;
    }

    void KSIntCalculator::PushDeupdateComponent()
    {
    	fStepNInteractions = 0;
    	fStepEnergyLoss = 0.0;
    }

    STATICINT sKSIntCalculatorDict =
		KSDictionary< KSIntCalculator >::AddComponent( &KSIntCalculator::GetStepNInteractions, "step_number_of_interactions" ) +
		KSDictionary< KSIntCalculator >::AddComponent( &KSIntCalculator::GetStepEnergyLoss, "step_energy_loss" );

} /* namespace Kassiopeia */


