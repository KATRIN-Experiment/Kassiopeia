#include "KSComponentDeltaBuilder.h"
#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSComponentDeltaBuilder::~KComplexElement()
    {
    }

    STATICINT sKSComponentDeltaStructure =
        KSComponentDeltaBuilder::Attribute< string >( "name" ) +
        KSComponentDeltaBuilder::Attribute< string >( "group" ) +
        KSComponentDeltaBuilder::Attribute< string >( "component" ) +
        KSComponentDeltaBuilder::Attribute< string >( "parent" );


    STATICINT sKSComponentDelta =
        KSComponentGroupBuilder::ComplexElement< KSComponentDeltaData >( "component_delta" ) +
        KSComponentGroupBuilder::ComplexElement< KSComponentDeltaData >( "output_delta" ) +
        KSRootBuilder::ComplexElement< KSComponentDeltaData >( "ks_component_delta" ) +
        KSRootBuilder::ComplexElement< KSComponentDeltaData >( "output_delta" );

}
