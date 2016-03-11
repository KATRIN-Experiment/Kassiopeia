#include "KSGenValueListBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueListBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenValueListStructure =
        KSGenValueListBuilder::Attribute< string >( "name" ) +
        KSGenValueListBuilder::Attribute< double >( "add_value" );

    STATICINT sKSGenValueList =
        KSRootBuilder::ComplexElement< KSGenValueList >( "ksgen_value_list" );

}
