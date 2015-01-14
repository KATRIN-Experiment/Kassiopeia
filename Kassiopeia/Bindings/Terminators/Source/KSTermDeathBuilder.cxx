#include "KSTermDeathBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermDeathBuilder::~KComplexElement()
    {
    }

    static int sKSTermDeathStructure =
        KSTermDeathBuilder::Attribute< string >( "name" );

    static int sKSTermDeath =
        KSRootBuilder::ComplexElement< KSTermDeath >( "ksterm_death" );


}
