#include "KSTermMaxRBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMaxRBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermMaxRStructure =
        KSTermMaxRBuilder::Attribute< string >( "name" ) +
        KSTermMaxRBuilder::Attribute< double >( "r" );

    STATICINT sKSTermMaxR =
        KSRootBuilder::ComplexElement< KSTermMaxR >( "ksterm_max_r" );

}
