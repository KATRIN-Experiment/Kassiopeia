#include "KSNavSpaceBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSNavSpaceBuilder::~KComplexElement()
    {
    }

    static int sKSNavSpaceStructure =
        KSNavSpaceBuilder::Attribute< string >( "name" ) +
        KSNavSpaceBuilder::Attribute< bool >( "enter_split" ) +
        KSNavSpaceBuilder::Attribute< bool >( "exit_split" ) +
        KSNavSpaceBuilder::Attribute< double >( "tolerance" );

    static int sToolboxKSNavSpace =
        KSRootBuilder::ComplexElement< KSNavSpace >( "ksnav_space" );

}
