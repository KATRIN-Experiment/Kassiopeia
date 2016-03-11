#include "KSGenTimeCompositeBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSRootBuilder.h"

#ifdef KASSIOPEIA_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#include "KSGenValueHistogramBuilder.h"
#endif

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenTimeCompositeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenTimeCompositeStructure =
        KSGenTimeCompositeBuilder::Attribute< string >( "name" ) +
        KSGenTimeCompositeBuilder::Attribute< string >( "time_value" ) +
        KSGenTimeCompositeBuilder::ComplexElement< KSGenValueFix >( "time_fix" ) +
        KSGenTimeCompositeBuilder::ComplexElement< KSGenValueSet >( "time_set" ) +
        KSGenTimeCompositeBuilder::ComplexElement< KSGenValueList >( "time_list" ) +
        KSGenTimeCompositeBuilder::ComplexElement< KSGenValueUniform >( "time_uniform" ) +
        KSGenTimeCompositeBuilder::ComplexElement< KSGenValueGauss >( "time_gauss" );

    STATICINT sKSGenTimeComposite =
        KSRootBuilder::ComplexElement< KSGenTimeComposite >( "ksgen_time_composite" );

#ifdef KASSIOPEIA_USE_ROOT
    STATICINT sKSGenTimeCompositeStructureROOT =
            KSGenTimeCompositeBuilder::ComplexElement< KSGenValueFormula >( "time_formula" ) +
            KSGenTimeCompositeBuilder::ComplexElement< KSGenValueHistogram >( "time_histogram" );
#endif

}
