#include "KSIntDecayBuilder.h"
#include "KSIntDecayCalculatorDeathConstRateBuilder.h"
#include "KSIntDecayCalculatorGlukhovSpontaneousBuilder.h"
#include "KSIntDecayCalculatorGlukhovDeExcitationBuilder.h"
#include "KSIntDecayCalculatorGlukhovExcitationBuilder.h"
#include "KSIntDecayCalculatorGlukhovIonisationBuilder.h"
#include "KSIntDecayCalculatorFerencBBRTransitionBuilder.h"
#include "KSIntDecayCalculatorFerencSpontaneousBuilder.h"
#include "KSIntDecayCalculatorFerencIonisationBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    KSIntDecayCalculatorSet::KSIntDecayCalculatorSet()
    {
    }
    KSIntDecayCalculatorSet::~KSIntDecayCalculatorSet()
    {
    }

    template< >
    KSIntDecayBuilder::~KComplexElement()
    {
    }   

    STATICINT sKSIntDecayStructure =
        KSIntDecayBuilder::Attribute< string >( "name" ) +
        KSIntDecayBuilder::Attribute< string >( "calculator" ) +
        KSIntDecayBuilder::Attribute< string >( "calculators" ) +
        KSIntDecayBuilder::Attribute< double >( "enhancement" ) +
        KSIntDecayBuilder::ComplexElement< KSIntDecayCalculatorDeathConstRate >( "decay_death_const_rate" )+
        KSIntDecayBuilder::ComplexElement< KSIntDecayCalculatorGlukhovSpontaneous >( "decay_glukhov_spontaneous" )+
        KSIntDecayBuilder::ComplexElement< KSIntDecayCalculatorGlukhovIonisation >( "decay_glukhov_ionisation" )+
        KSIntDecayBuilder::ComplexElement< KSIntDecayCalculatorGlukhovExcitation >( "decay_glukhov_excitation" )+
        KSIntDecayBuilder::ComplexElement< KSIntDecayCalculatorGlukhovDeExcitation >( "decay_glukhov_deexcitation" ) +
        KSIntDecayBuilder::ComplexElement< KSIntDecayCalculatorFerencBBRTransition >( "decay_ferenc_bbr" ) +
        KSIntDecayBuilder::ComplexElement< KSIntDecayCalculatorFerencSpontaneous >( "decay_ferenc_spontaneous" )+
        KSIntDecayBuilder::ComplexElement< KSIntDecayCalculatorFerencIonisation >( "decay_ferenc_ionisation" );

    STATICINT sKSIntDecay =
        KSRootBuilder::ComplexElement< KSIntDecay >( "ksint_decay" );

}
