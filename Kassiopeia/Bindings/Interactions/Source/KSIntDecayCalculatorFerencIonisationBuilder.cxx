//
// Created by trost on 03.06.15.
//

#include "KSIntDecayCalculatorFerencIonisationBuilder.h"
#include "KSGenGeneratorCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSIntDecayCalculatorFerencIonisationBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntDecayCalculatorFerencIonisationBuilderStructure =
            KSIntDecayCalculatorFerencIonisationBuilder::Attribute< string >( "name" ) +
            KSIntDecayCalculatorFerencIonisationBuilder::Attribute< long long >( "target_pid" ) +
            KSIntDecayCalculatorFerencIonisationBuilder::Attribute< long long >( "min_pid" ) +
            KSIntDecayCalculatorFerencIonisationBuilder::Attribute< long long >( "max_pid" ) +
            KSIntDecayCalculatorFerencIonisationBuilder::Attribute< double >( "temperature" ) +
            KSIntDecayCalculatorFerencIonisationBuilder::ComplexElement< KSGenGeneratorComposite >( "decay_product_generator");

    STATICINT sToolboxKSIntDecayCalculatorFerencIonisation =
            KSRootBuilder::ComplexElement< KSIntDecayCalculatorFerencIonisation >( "ksint_decay_calculator_ferenc_ionisation" );
}
