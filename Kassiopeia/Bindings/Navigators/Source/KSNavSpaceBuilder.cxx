#include "KSNavSpaceBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSNavSpaceBuilder::~KComplexElement()
    {
    }

    STATICINT sKSNavSpaceStructure =
        KSNavSpaceBuilder::Attribute< string >( "name" ) +
        KSNavSpaceBuilder::Attribute< bool >( "enter_split" ) +
        KSNavSpaceBuilder::Attribute< bool >( "exit_split" ) +
        KSNavSpaceBuilder::Attribute< bool >( "fail_check" ) +
        KSNavSpaceBuilder::Attribute< double >( "tolerance" );

    STATICINT sToolboxKSNavSpace =
        KSRootBuilder::ComplexElement< KSNavSpace >( "ksnav_space" );

}
