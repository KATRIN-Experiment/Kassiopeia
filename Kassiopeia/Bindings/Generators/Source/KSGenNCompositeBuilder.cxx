#include "KSGenNCompositeBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueParetoBuilder.h"
#include "KSRootBuilder.h"

#ifdef KASSIOPEIA_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#endif

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenNCompositeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenNCompositeStructure =
        KSGenNCompositeBuilder::Attribute< string >( "name" ) +
        KSGenNCompositeBuilder::Attribute< string >( "n_value" ) +
        KSGenNCompositeBuilder::ComplexElement< KSGenValueFix >( "n_fix" ) +
        KSGenNCompositeBuilder::ComplexElement< KSGenValueSet >( "n_set" ) +
        KSGenNCompositeBuilder::ComplexElement< KSGenValueList >( "n_list" ) +
        KSGenNCompositeBuilder::ComplexElement< KSGenValueUniform >( "n_uniform" ) +
        KSGenNCompositeBuilder::ComplexElement< KSGenValueGauss >( "n_gauss" )+
        KSGenNCompositeBuilder::ComplexElement< KSGenValuePareto >( "n_pareto" );

    STATICINT sKSGenNComposite =
        KSRootBuilder::ComplexElement< KSGenNComposite >( "ksgen_n_composite" );

#ifdef KASSIOPEIA_USE_ROOT
    STATICINT sKSGenNCompositeStructureROOT =
            KSGenNCompositeBuilder::ComplexElement< KSGenValueFormula >( "n_formula" );
#endif

}
