#include "KSTermMaxLengthBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSTermMaxLengthBuilder::~KComplexElement()
    {
    }

    STATICINT sKSTermMaxLengthStructure =
        KSTermMaxLengthBuilder::Attribute< string >( "name" ) +
        KSTermMaxLengthBuilder::Attribute< double >( "length" );

    STATICINT sKSTermMaxLength =
        KSRootBuilder::ComplexElement< KSTermMaxLength >( "ksterm_max_length" );

}
