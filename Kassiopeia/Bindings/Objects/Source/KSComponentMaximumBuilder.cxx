#include "KSComponentMaximumBuilder.h"
#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSComponentMaximumBuilder::~KComplexElement()
    {
    }

    static int sKSComponentMaximumStructure =
        KSComponentMaximumBuilder::Attribute< string >( "name" ) +
        KSComponentMaximumBuilder::Attribute< string >( "group" ) +
        KSComponentMaximumBuilder::Attribute< string >( "component" );

    static int sKSComponentMaximum =
        KSComponentGroupBuilder::ComplexElement< KSComponentMaximumData >( "component_maximum" ) +
        KSRootBuilder::ComplexElement< KSComponentMaximumData >( "ks_component_maximum" );

}
