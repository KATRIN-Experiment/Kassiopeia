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
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("name") +
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("surface") +
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("space") +
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("x") +
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("y") +
    KSGenPositionRectangularCompositeBuilder::Attribute<std::string>("z") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFix>("x_fix") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueSet>("x_set") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueList>("x_list") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueUniform>("x_uniform") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueGauss>("x_gauss") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFix>("y_fix") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueSet>("y_set") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueList>("y_list") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueUniform>("y_uniform") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueGauss>("y_gauss") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFix>("z_fix") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueSet>("z_set") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueList>("z_list") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueUniform>("z_uniform") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueGauss>("z_gauss");

STATICINT sToolboxKSGenPositionRectangularComposite =
    KSRootBuilder::ComplexElement<KSGenPositionRectangularComposite>("ksgen_position_rectangular_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenPositionRectangularCompositeStructureROOT =
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFormula>("x_formula") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueHistogram>("x_histogram") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFormula>("y_formula") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueHistogram>("y_histogram") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueFormula>("z_formula") +
    KSGenPositionRectangularCompositeBuilder::ComplexElement<KSGenValueHistogram>("z_histogram");
#endif

}  // namespace katrin
