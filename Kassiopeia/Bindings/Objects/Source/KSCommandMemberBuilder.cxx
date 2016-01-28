#include "KSCommandMemberBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSCommandMemberBuilder::~KComplexElement()
    {
    }

    STATICINT sKSCommandStructure =
        KSCommandMemberBuilder::Attribute< string >( "name" ) +
        KSCommandMemberBuilder::Attribute< string >( "parent" ) +
        KSCommandMemberBuilder::Attribute< string >( "child" ) +
        KSCommandMemberBuilder::Attribute< string >( "field" );

    STATICINT sKSCommand =
        KSRootBuilder::ComplexElement< KSCommandMemberData >( "ks_command_member" );

}
