#include "KSTermMaxTimeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMaxTimeBuilder::~KComplexElement()
    {
    }

    static int sKSTermMaxTimeStructure =
        KSTermMaxTimeBuilder::Attribute< string >( "name" ) +
        KSTermMaxTimeBuilder::Attribute< double >( "time" );

    static int sKSTermMaxTime =
        KSRootBuilder::ComplexElement< KSTermMaxTime >( "ksterm_max_time" );

}
