#include "KSIntDecayCalculatorGlukhovExcitationBuilder.h"
#include "KSGenGeneratorCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntDecayCalculatorGlukhovExcitationBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntDecayCalculatorGlukhovExcitationBuilderStructure =
        KSIntDecayCalculatorGlukhovExcitationBuilder::Attribute< string >( "name" ) +
        KSIntDecayCalculatorGlukhovExcitationBuilder::Attribute< long long >( "target_pid" ) +
        KSIntDecayCalculatorGlukhovExcitationBuilder::Attribute< long long >( "min_pid" ) +
        KSIntDecayCalculatorGlukhovExcitationBuilder::Attribute< long long >( "max_pid" ) +
        KSIntDecayCalculatorGlukhovExcitationBuilder::Attribute< double >( "temperature" ) +
        KSIntDecayCalculatorGlukhovExcitationBuilder::ComplexElement< KSGenGeneratorComposite >( "decay_product_generator");

    STATICINT sToolboxKSIntDecayCalculatorGlukhovExcitation =
        KSRootBuilder::ComplexElement< KSIntDecayCalculatorGlukhovExcitation >( "ksint_decay_calculator_glukhov_excitation" );
}
