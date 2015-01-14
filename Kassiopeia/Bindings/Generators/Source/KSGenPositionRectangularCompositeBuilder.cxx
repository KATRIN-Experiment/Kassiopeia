#include "KSGenPositionRectangularCompositeBuilder.h"
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
    KSGenPositionRectangularCompositeBuilder::~KComplexElement()
    {
    }

    static int sKSGenPositionRectangularCompositeStructure =
        KSGenPositionRectangularCompositeBuilder::Attribute< string >( "name" ) +
        KSGenPositionRectangularCompositeBuilder::Attribute< string >( "surface" ) +
        KSGenPositionRectangularCompositeBuilder::Attribute< string >( "space" ) +
        KSGenPositionRectangularCompositeBuilder::Attribute< string >( "x" ) +
        KSGenPositionRectangularCompositeBuilder::Attribute< string >( "y" ) +
        KSGenPositionRectangularCompositeBuilder::Attribute< string >( "z" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueFix >( "x_fix" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueSet >( "x_set" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueList >( "x_list" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueUniform >( "x_uniform" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueGauss >( "x_gauss" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueFix >( "y_fix" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueSet >( "y_set" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueList >( "y_list" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueUniform >( "y_uniform" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueGauss >( "y_gauss" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueFix >( "z_fix" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueSet >( "z_set" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueList >( "z_list" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueUniform >( "z_uniform" ) +
        KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueGauss >( "z_gauss" );

    static int sToolboxKSGenPositionRectangularComposite =
        KSRootBuilder::ComplexElement< KSGenPositionRectangularComposite >( "ksgen_position_rectangular_composite" );

#ifdef KASSIOPEIA_USE_ROOT
    static int sKSGenPositionRectangularCompositeStructureROOT =
            KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueFormula >( "x_formula" ) +
            KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueFormula >( "y_formula" ) +
            KSGenPositionRectangularCompositeBuilder::ComplexElement< KSGenValueFormula >( "z_formula" );
#endif

}
