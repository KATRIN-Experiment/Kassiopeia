#include "KSGenValueSetBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueSetBuilder::~KComplexElement()
    {
    }

    static int sKSGenValueSetStructure =
        KSGenValueSetBuilder::Attribute< string >( "name" ) +
        KSGenValueSetBuilder::Attribute< double >( "value_start" ) +
        KSGenValueSetBuilder::Attribute< double >( "value_stop" ) +
        KSGenValueSetBuilder::Attribute< unsigned int >( "value_count" );

    static int sKSGenValueSet =
        KSRootBuilder::ComplexElement< KSGenValueSet >( "ksgen_value_set" );

}
