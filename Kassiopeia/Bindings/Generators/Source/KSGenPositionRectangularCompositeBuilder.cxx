#include "KSGenPositionRectangularCompositeBuilder.h"

#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSRootBuilder.h"

#ifdef Kassiopeia_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#include "KSGenValueHistogramBuilder.h"
#endif

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenPositionRectangularCompositeBuilder::~KComplexElement() = default;

STATICINT sKSGenPositionRectangularCompositeStructure =
    //! the name of this position creator.
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("name") +
    //! the name of a surface in kgeobag to use as the origin of coordinates.
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("surface") +
    //! the name of a space in kgeobag to use as the origin of coordinates.
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("space") +
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("x") +
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("y") +
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("z") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFix>("x_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueSet>("x_set") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueList>("x_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueUniform>("x_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueGauss>("x_gauss") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFix>("y_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueSet>("y_set") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueList>("y_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueUniform>("y_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueGauss>("y_gauss") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFix>("z_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueSet>("z_set") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueList>("z_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueUniform>("z_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueGauss>("z_gauss");

STATICINT sToolboxKSGenPositionRectangularComposite =
    KSRootBuilder::ComplexElement<KSGenPositionRectangularComposite>("ksgen_position_rectangular_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenPositionRectangularCompositeStructureROOT =
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFormula>("x_formula") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueHistogram>("x_histogram") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFormula>("y_formula") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueHistogram>("y_histogram") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (x|y|z) coordinate.
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFormula>("z_formula") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueHistogram>("z_histogram");
#endif

}  // namespace katrin
