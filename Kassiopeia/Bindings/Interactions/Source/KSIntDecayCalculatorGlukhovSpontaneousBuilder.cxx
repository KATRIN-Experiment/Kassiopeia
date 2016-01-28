#include "KSIntDecayCalculatorGlukhovSpontaneousBuilder.h"
#include "KSGenGeneratorCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSIntDecayCalculatorGlukhovSpontaneousBuilder::~KComplexElement()
    {
    }

    STATICINT sKSIntDecayCalculatorGlukhovSpontaneousBuilderStructure =
        KSIntDecayCalculatorGlukhovSpontaneousBuilder::Attribute< string >( "name" ) +
        KSIntDecayCalculatorGlukhovSpontaneousBuilder::Attribute< long long >( "target_pid" ) +
        KSIntDecayCalculatorGlukhovSpontaneousBuilder::Attribute< long long >( "min_pid" ) +
        KSIntDecayCalculatorGlukhovSpontaneousBuilder::Attribute< long long >( "max_pid" ) +
        KSIntDecayCalculatorGlukhovSpontaneousBuilder::ComplexElement< KSGenGeneratorComposite >( "decay_product_generator");

    STATICINT sToolboxKSIntDecayCalculatorGlukhovSpontaneous =
        KSRootBuilder::ComplexElement< KSIntDecayCalculatorGlukhovSpontaneous >( "ksint_decay_calculator_glukhov_spontaneous" );
}
