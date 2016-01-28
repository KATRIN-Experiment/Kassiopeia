#include "KSTermDeathBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermDeathBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermDeathStructure =
        KSTermDeathBuilder::Attribute< string >( "name" );

    STATICINT sKSTermDeath =
        KSRootBuilder::ComplexElement< KSTermDeath >( "ksterm_death" );


}
