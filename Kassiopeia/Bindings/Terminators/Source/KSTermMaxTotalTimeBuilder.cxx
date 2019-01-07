#include "KSTermMaxTotalTimeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTermMaxTotalTimeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermMaxTotalTimeStructure =
        KSTermMaxTotalTimeBuilder::Attribute< string >( "name" ) +
        KSTermMaxTotalTimeBuilder::Attribute< double >( "time" );

    STATICINT sKSTermMaxStepTime =
        KSRootBuilder::ComplexElement< KSTermMaxTotalTime >( "ksterm_max_total_time" );

}
