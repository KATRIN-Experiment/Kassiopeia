#include "KSGenValueListBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueListBuilder::~KComplexElement()
    {
    }

    static int sKSGenValueListStructure =
        KSGenValueListBuilder::Attribute< string >( "name" ) +
        KSGenValueListBuilder::Attribute< double >( "add_value" );

    static int sKSGenValueList =
        KSRootBuilder::ComplexElement< KSGenValueList >( "ksgen_value_list" );

}
