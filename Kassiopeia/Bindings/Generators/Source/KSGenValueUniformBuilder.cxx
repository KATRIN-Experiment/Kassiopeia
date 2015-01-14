#include "KSGenValueUniformBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueUniformBuilder::~KComplexElement()
    {
    }

    static int sKSGenValueUniformStructure =
        KSGenValueUniformBuilder::Attribute< string >( "name" ) +
        KSGenValueUniformBuilder::Attribute< double >( "value_min" ) +
        KSGenValueUniformBuilder::Attribute< double >( "value_max" );

    static int sToolboxKSGenValueUniform =
        KSRootBuilder::ComplexElement< KSGenValueUniform >( "ksgen_value_uniform" );

}
