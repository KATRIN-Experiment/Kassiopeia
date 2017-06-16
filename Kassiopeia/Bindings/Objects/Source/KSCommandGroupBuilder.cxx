#include "KSCommandGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSCommandGroupBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGroupStructure =
        KSCommandGroupBuilder::Attribute< string >( "name" ) +
        KSCommandGroupBuilder::Attribute< string >( "command" );

    STATICINT sKSGroup =
        KSCommandGroupBuilder::ComplexElement< KSCommandGroup >( "command_group" ) +
        KSRootBuilder::ComplexElement< KSCommandGroup >( "ks_command_group" );

}
