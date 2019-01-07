#include "KSGenSpinRelativeCompositeBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueGeneralizedGaussBuilder.h"
#include "KSGenValueAngleCosineBuilder.h"
#include "KSGenValueAngleSphericalBuilder.h"
#include "KSRootBuilder.h"

#ifdef Kassiopeia_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#include "KSGenValueHistogramBuilder.h"
#endif

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

    template< >
    KSGenSpinRelativeCompositeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenSpinRelativeCompositeStructure =
        KSGenSpinRelativeCompositeBuilder::Attribute< string >( "name" ) +
        KSGenSpinRelativeCompositeBuilder::Attribute< string >( "theta" ) +
        KSGenSpinRelativeCompositeBuilder::Attribute< string >( "phi" ) +
        KSGenSpinRelativeCompositeBuilder::Attribute< string >( "surface" ) +
        KSGenSpinRelativeCompositeBuilder::Attribute< string >( "space" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueFix >( "theta_fix" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueSet >( "theta_set" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueList >( "theta_list" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueUniform >( "theta_uniform" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueGauss >( "theta_gauss" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueGeneralizedGauss >( "theta_generalized_gauss" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueAngleCosine >( "theta_cosine" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueAngleSpherical >( "theta_spherical" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueFix >( "phi_fix" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueSet >( "phi_set" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueList >( "phi_list" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueUniform >( "phi_uniform" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueGauss >( "phi_gauss" ) +
        KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueGeneralizedGauss >( "phi_generalized_gauss" );

    STATICINT sKSGenSpinRelativeComposite =
        KSRootBuilder::ComplexElement< KSGenSpinRelativeComposite >( "ksgen_spin_composite_relative" );

#ifdef Kassiopeia_USE_ROOT
    STATICINT sKSGenSpinRelativeCompositeStructureROOT =
            KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueFormula >( "theta_formula" ) +
            KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueHistogram >( "theta_histogram" ) +
            KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueFormula >( "phi_formula" ) +
            KSGenSpinRelativeCompositeBuilder::ComplexElement< KSGenValueHistogram >( "phi_histogram" );
#endif

}
