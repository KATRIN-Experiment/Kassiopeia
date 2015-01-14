#include "KSTermMaxZBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMaxZBuilder::~KComplexElement()
    {
    }

    static int sKSTermMaxZStructure =
        KSTermMaxZBuilder::Attribute< string >( "name" ) +
        KSTermMaxZBuilder::Attribute< double >( "z" );

    static int sToolboxKSTermMaxZ =
        KSRootBuilder::ComplexElement< KSTermMaxZ >( "ksterm_max_z" );


}
