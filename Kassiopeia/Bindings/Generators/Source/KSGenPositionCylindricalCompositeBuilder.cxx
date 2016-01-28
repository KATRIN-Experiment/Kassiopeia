#include "KSGenPositionCylindricalCompositeBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueRadiusCylindricalBuilder.h"
#include "KSRootBuilder.h"

#ifdef Kassiopeia_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#include "KSGenValueHistogramBuilder.h"
#endif

using namespace Kassiopeia;
namespace katrin
{

    template< >
    KSGenPositionCylindricalCompositeBuilder::~KComplexElement()
    {
    }

    STATICINT sKSGenPositionCylindricalCompositeStructure =
        KSGenPositionCylindricalCompositeBuilder::Attribute< string >( "name" ) +
        KSGenPositionCylindricalCompositeBuilder::Attribute< string >( "surface" ) +
        KSGenPositionCylindricalCompositeBuilder::Attribute< string >( "space" ) +
        KSGenPositionCylindricalCompositeBuilder::Attribute< string >( "r" ) +
        KSGenPositionCylindricalCompositeBuilder::Attribute< string >( "phi" ) +
        KSGenPositionCylindricalCompositeBuilder::Attribute< string >( "z" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueFix >( "r_fix" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueSet >( "r_set" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueList >( "r_list" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueUniform >( "r_uniform" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueGauss >( "r_gauss" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueRadiusCylindrical >( "r_cylindrical" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueFix >( "phi_fix" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueSet >( "phi_set" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueList >( "phi_list" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueUniform >( "phi_uniform" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueGauss >( "phi_gauss" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueFix >( "z_fix" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueSet >( "z_set" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueList >( "z_list" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueUniform >( "z_uniform" ) +
        KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueGauss >( "z_gauss" );

    STATICINT sKSGenPositionCylindricalComposite =
        KSRootBuilder::ComplexElement< KSGenPositionCylindricalComposite >( "ksgen_position_cylindrical_composite" );

#ifdef Kassiopeia_USE_ROOT
    STATICINT sKSGenPositionCylindricalCompositeStructureROOT =
            KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueFormula >( "r_formula" ) +
            KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueHistogram >( "r_histogram" ) +KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueFormula >( "phi_formula" ) +KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueHistogram >( "phi_histogram" ) +
            KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueFormula >( "z_formula" ) +KSGenPositionCylindricalCompositeBuilder::ComplexElement< KSGenValueHistogram >( "z_histogram" );
#endif

}
