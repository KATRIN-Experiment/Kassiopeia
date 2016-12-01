#include "KSComponentMaximumBuilder.h"
#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSComponentMaximumBuilder::~KComplexElement()
    {
    }

    STATICINT sKSComponentMaximumStructure =
        KSComponentMaximumBuilder::Attribute< string >( "name" ) +
        KSComponentMaximumBuilder::Attribute< string >( "group" ) +
        KSComponentMaximumBuilder::Attribute< string >( "component" ) +
        KSComponentMaximumBuilder::Attribute< string >( "parent" );

    STATICINT sKSComponentMaximum =
        KSComponentGroupBuilder::ComplexElement< KSComponentMaximumData >( "component_maximum" ) +
        KSComponentGroupBuilder::ComplexElement< KSComponentMaximumData >( "output_maximum" ) +
        KSRootBuilder::ComplexElement< KSComponentMaximumData >( "ks_component_maximum" ) +
        KSRootBuilder::ComplexElement< KSComponentMaximumData >( "output_maximum" );

}
