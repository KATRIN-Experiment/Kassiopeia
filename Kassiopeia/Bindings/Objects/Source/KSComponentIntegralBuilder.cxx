#include "KSComponentIntegralBuilder.h"
#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSComponentIntegralBuilder::~KComplexElement()
    {
    }

    STATICINT sKSComponentIntegralStructure =
        KSComponentIntegralBuilder::Attribute< string >( "name" ) +
        KSComponentIntegralBuilder::Attribute< string >( "group" ) +
        KSComponentIntegralBuilder::Attribute< string >( "component" ) +
        KSComponentIntegralBuilder::Attribute< string >( "parent" );

    STATICINT sKSComponentIntegral =
        KSComponentGroupBuilder::ComplexElement< KSComponentIntegralData >( "component_integral" ) +
        KSComponentGroupBuilder::ComplexElement< KSComponentIntegralData >( "output_integral" ) +
        KSRootBuilder::ComplexElement< KSComponentIntegralData >( "ks_component_integral" ) +
        KSRootBuilder::ComplexElement< KSComponentIntegralData >( "output_integral" );


}
