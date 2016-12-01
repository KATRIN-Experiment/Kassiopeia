#include "KSIntDecayCalculatorGlukhovIonisationBuilder.h"
#include "KSGenGeneratorCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSIntDecayCalculatorGlukhovIonisationBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntDecayCalculatorGlukhovIonisationBuilderStructure =
        KSIntDecayCalculatorGlukhovIonisationBuilder::Attribute< string >( "name" ) +
        KSIntDecayCalculatorGlukhovIonisationBuilder::Attribute< long long >( "target_pid" ) +
        KSIntDecayCalculatorGlukhovIonisationBuilder::Attribute< long long >( "min_pid" ) +
        KSIntDecayCalculatorGlukhovIonisationBuilder::Attribute< long long >( "max_pid" ) +
        KSIntDecayCalculatorGlukhovIonisationBuilder::Attribute< double >( "temperature" ) +
        KSIntDecayCalculatorGlukhovIonisationBuilder::ComplexElement< KSGenGeneratorComposite >( "decay_product_generator");

    STATICINT sToolboxKSIntDecayCalculatorGlukhovIonisation =
        KSRootBuilder::ComplexElement< KSIntDecayCalculatorGlukhovIonisation >( "ksint_decay_calculator_glukhov_ionisation" );
}
