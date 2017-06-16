#include "KSGenValueGeneralizedGaussBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSGenValueGeneralizedGaussBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenValueGeneralizedGaussStructure =
        KSGenValueGeneralizedGaussBuilder::Attribute< string >( "name" ) +
        KSGenValueGeneralizedGaussBuilder::Attribute< double >( "value_min" ) +
        KSGenValueGeneralizedGaussBuilder::Attribute< double >( "value_max" ) +
        KSGenValueGeneralizedGaussBuilder::Attribute< double >( "value_mean" ) +
        KSGenValueGeneralizedGaussBuilder::Attribute< double >( "value_sigma" ) +
        KSGenValueGeneralizedGaussBuilder::Attribute< double >( "value_skew" );

    STATICINT sKSGenValueGeneralizedGauss =
        KSRootBuilder::ComplexElement< KSGenValueGeneralizedGauss >( "ksgen_value_generalized_gauss" );

}
