#include "KSComponentMinimumBuilder.h"
#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSComponentMinimumBuilder::~KComplexElement()
    {
    }

    STATICINT sKSComponentMinimumStructure =
        KSComponentMinimumBuilder::Attribute< string >( "name" ) +
        KSComponentMinimumBuilder::Attribute< string >( "group" ) +
        KSComponentMinimumBuilder::Attribute< string >( "component" ) +
        KSComponentMinimumBuilder::Attribute< string >( "parent" );

    STATICINT sKSComponentMinimum =
        KSComponentGroupBuilder::ComplexElement< KSComponentMinimumData >( "component_minimum" ) +
        KSComponentGroupBuilder::ComplexElement< KSComponentMinimumData >( "output_minimum" ) +
        KSRootBuilder::ComplexElement< KSComponentMinimumData >( "ks_component_minimum" ) +
        KSRootBuilder::ComplexElement< KSComponentMinimumData >( "output_minimum" );

}
