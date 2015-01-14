#include "KSCommandMemberBuilder.h"
#include "KSCommandGroupBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSCommandMemberBuilder::~KComplexElement()
    {
    }

    static int sKSCommandStructure =
        KSCommandMemberBuilder::Attribute< string >( "name" ) +
        KSCommandMemberBuilder::Attribute< string >( "parent" ) +
        KSCommandMemberBuilder::Attribute< string >( "child" ) +
        KSCommandMemberBuilder::Attribute< string >( "field" );

    static int sKSCommand =
        KSCommandGroupBuilder::ComplexElement< KSCommandMemberData >( "command_member" ) +
        KSRootBuilder::ComplexElement< KSCommandMemberData >( "ks_command_member" );

}
