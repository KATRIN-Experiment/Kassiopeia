#include "KSComponentMathBuilder.h"
#include "KSComponentGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSComponentMathBuilder::~KComplexElement()
    {
    }

    STATICINT sKSComponentMathStructure =
		KSComponentMathBuilder::Attribute< string >( "name" ) +
		KSComponentMathBuilder::Attribute< string >( "group" ) +
		KSComponentMathBuilder::Attribute< string >( "component" ) +
        KSComponentMathBuilder::Attribute< string >( "parent" ) +
		KSComponentMathBuilder::Attribute< string >( "term" );

    STATICINT sKSComponentMath =
        KSComponentGroupBuilder::ComplexElement< KSComponentMathData >( "component_math" ) +
        KSComponentGroupBuilder::ComplexElement< KSComponentMathData >( "output_math" ) +
        KSRootBuilder::ComplexElement< KSComponentMathData >( "ks_component_math" )+
        KSRootBuilder::ComplexElement< KSComponentMathData >( "output_math" );

}
