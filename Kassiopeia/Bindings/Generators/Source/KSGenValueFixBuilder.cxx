#include "KSGenValueFixBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueFixBuilder::~KComplexElement()
    {
    }

    static int sKSGenValueFixStructure =
        KSGenValueFixBuilder::Attribute< string >( "name" ) +
        KSGenValueFixBuilder::Attribute< double >( "value" );

    static int sKSGenValueFix =
        KSRootBuilder::ComplexElement< KSGenValueFix >( "ksgen_value_fix" );

}
