#include "KSComponentIntegralBuilder.h"
#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSComponentIntegralBuilder::~KComplexElement()
    {
    }

    static int sKSComponentIntegralStructure =
        KSComponentIntegralBuilder::Attribute< string >( "name" ) +
        KSComponentIntegralBuilder::Attribute< string >( "group" ) +
        KSComponentIntegralBuilder::Attribute< string >( "component" );

    static int sKSComponentIntegral =
        KSComponentGroupBuilder::ComplexElement< KSComponentIntegralData >( "component_integral" ) +
        KSRootBuilder::ComplexElement< KSComponentIntegralData >( "ks_component_integral" );

}
