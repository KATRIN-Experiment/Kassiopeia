#include "KSGenValueGaussBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenValueGaussBuilder::~KComplexElement()
    {
    }

    static int sKSGenValueGaussStructure =
        KSGenValueGaussBuilder::Attribute< string >( "name" ) +
        KSGenValueGaussBuilder::Attribute< double >( "value_min" ) +
        KSGenValueGaussBuilder::Attribute< double >( "value_max" ) +
        KSGenValueGaussBuilder::Attribute< double >( "value_mean" ) +
        KSGenValueGaussBuilder::Attribute< double >( "value_sigma" );

    static int sKSGenValueGauss =
        KSRootBuilder::ComplexElement< KSGenValueGauss >( "ksgen_value_gauss" );

}
