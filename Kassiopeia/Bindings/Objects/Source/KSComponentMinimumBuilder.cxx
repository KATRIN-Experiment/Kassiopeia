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

    static int sKSComponentMinimumStructure =
        KSComponentMinimumBuilder::Attribute< string >( "name" ) +
        KSComponentMinimumBuilder::Attribute< string >( "group" ) +
        KSComponentMinimumBuilder::Attribute< string >( "component" );

    static int sKSComponentMinimum =
        KSComponentGroupBuilder::ComplexElement< KSComponentMinimumData >( "component_minimum" ) +
        KSRootBuilder::ComplexElement< KSComponentMinimumData >( "ks_component_minimum" );

}
