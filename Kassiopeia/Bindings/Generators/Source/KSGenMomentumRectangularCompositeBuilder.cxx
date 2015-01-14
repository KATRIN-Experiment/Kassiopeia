#include "KSGenMomentumRectangularCompositeBuilder.h"
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
    KSGenMomentumRectangularCompositeBuilder::~KComplexElement()
    {
    }

    static int sKSGenMomentumRectangularCompositeStructure =
        KSGenMomentumRectangularCompositeBuilder::Attribute< string >( "name" ) +
        KSGenMomentumRectangularCompositeBuilder::Attribute< string >( "surface" ) +
        KSGenMomentumRectangularCompositeBuilder::Attribute< string >( "space" ) +
        KSGenMomentumRectangularCompositeBuilder::Attribute< string >( "x" ) +
        KSGenMomentumRectangularCompositeBuilder::Attribute< string >( "y" ) +
        KSGenMomentumRectangularCompositeBuilder::Attribute< string >( "z" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueFix >( "x_fix" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueSet >( "x_set" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueList >( "x_list" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueUniform >( "x_uniform" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueGauss >( "x_gauss" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueFix >( "y_fix" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueSet >( "y_set" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueList >( "y_list" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueUniform >( "y_uniform" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueGauss >( "y_gauss" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueFix >( "z_fix" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueSet >( "z_set" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueList >( "z_list" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueUniform >( "z_uniform" ) +
        KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueGauss >( "z_gauss" );

    static int sToolboxKSGenMomentumRectangularComposite =
        KSRootBuilder::ComplexElement< KSGenMomentumRectangularComposite >( "ksgen_momentum_rectangular_composite" );

#ifdef KASSIOPEIA_USE_ROOT
    static int sKSGenPositionRectangularCompositeStructureROOT =
            KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueFormula >( "x_formula" ) +
            KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueFormula >( "y_formula" ) +
            KSGenMomentumRectangularCompositeBuilder::ComplexElement< KSGenValueFormula >( "z_formula" );
#endif

}
