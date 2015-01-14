#include "KSTermMaxLengthBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSTermMaxLengthBuilder::~KComplexElement()
    {
    }

    static int sKSTermMaxLengthStructure =
        KSTermMaxLengthBuilder::Attribute< string >( "name" ) +
        KSTermMaxLengthBuilder::Attribute< double >( "length" );

    static int sKSTermMaxLength =
        KSRootBuilder::ComplexElement< KSTermMaxLength >( "ksterm_max_length" );

}
