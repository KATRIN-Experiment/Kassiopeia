#include "KSRootTerminatorBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootTerminatorBuilder::~KComplexElement()
    {
    }

    STATICINT sKSRootTerminator =
        KSRootBuilder::ComplexElement< KSRootTerminator >( "ks_root_terminator" );

    STATICINT sKSRootTerminatorStructure =
        KSRootTerminatorBuilder::Attribute< string >( "name" ) +
        KSRootTerminatorBuilder::Attribute< string >( "add_terminator" );

}
