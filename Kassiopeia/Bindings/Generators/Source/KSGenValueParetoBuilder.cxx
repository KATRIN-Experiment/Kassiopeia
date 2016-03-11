#include "KSGenValueParetoBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueParetoBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenValueParetoStructure =
        KSGenValueParetoBuilder::Attribute< string >( "name" ) +
        KSGenValueParetoBuilder::Attribute< double >( "value_min" ) +
        KSGenValueParetoBuilder::Attribute< double >( "value_max" ) +
        KSGenValueParetoBuilder::Attribute< double >( "slope" )+
        KSGenValueParetoBuilder::Attribute< double >( "cutoff" )+
        KSGenValueParetoBuilder::Attribute< double >( "offset" );

    STATICINT sKSGenValuePareto =
        KSRootBuilder::ComplexElement< KSGenValuePareto >( "ksgen_value_pareto" );

}
