#include "KSRootTerminatorBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSRootTerminatorBuilder::~KComplexElement()
    {
    }

    static int sKSRootTerminator =
        KSRootBuilder::ComplexElement< KSRootTerminator >( "ks_root_terminator" );

    static int sKSRootTerminatorStructure =
        KSRootTerminatorBuilder::Attribute< string >( "name" ) +
        KSRootTerminatorBuilder::Attribute< string >( "add_terminator" );

}
