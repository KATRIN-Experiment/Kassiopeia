#include "KSCommandGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSCommandGroupBuilder::~KComplexElement()
    {
    }

    static int sKSGroupStructure =
        KSCommandGroupBuilder::Attribute< string >( "name" ) +
        KSCommandGroupBuilder::Attribute< string >( "command" );

    static int sKSGroup =
        KSCommandGroupBuilder::ComplexElement< KSCommandGroup >( "command_group" ) +
        KSRootBuilder::ComplexElement< KSCommandGroup >( "ks_command_group" );

}
