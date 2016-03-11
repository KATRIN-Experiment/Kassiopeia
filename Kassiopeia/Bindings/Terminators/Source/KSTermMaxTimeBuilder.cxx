#include "KSTermMaxTimeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMaxTimeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermMaxTimeStructure =
        KSTermMaxTimeBuilder::Attribute< string >( "name" ) +
        KSTermMaxTimeBuilder::Attribute< double >( "time" );

    STATICINT sKSTermMaxTime =
        KSRootBuilder::ComplexElement< KSTermMaxTime >( "ksterm_max_time" );

}
