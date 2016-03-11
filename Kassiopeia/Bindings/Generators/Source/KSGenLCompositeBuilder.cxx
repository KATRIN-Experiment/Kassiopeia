#include "KSGenLCompositeBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSRootBuilder.h"

#ifdef KASSIOPEIA_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#endif

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenLCompositeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenLCompositeStructure =
        KSGenLCompositeBuilder::Attribute< string >( "name" ) +
        KSGenLCompositeBuilder::Attribute< string >( "l_value" ) +
        KSGenLCompositeBuilder::ComplexElement< KSGenValueFix >( "l_fix" ) +
        KSGenLCompositeBuilder::ComplexElement< KSGenValueSet >( "l_set" ) +
        KSGenLCompositeBuilder::ComplexElement< KSGenValueList >( "l_list" ) +
        KSGenLCompositeBuilder::ComplexElement< KSGenValueUniform >( "l_uniform" ) +
        KSGenLCompositeBuilder::ComplexElement< KSGenValueGauss >( "l_gauss" );

    STATICINT sKSGenLComposite =
        KSRootBuilder::ComplexElement< KSGenLComposite >( "ksgen_l_composite" );

#ifdef KASSIOPEIA_USE_ROOT
    STATICINT sKSGenLCompositeStructureROOT =
            KSGenLCompositeBuilder::ComplexElement< KSGenValueFormula >( "l_formula" );
#endif

}
