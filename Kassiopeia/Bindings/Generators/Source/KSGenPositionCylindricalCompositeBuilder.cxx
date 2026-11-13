#include "KSGenPositionCylindricalCompositeBuilder.h"

#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueRadiusCylindricalBuilder.h"
#include "KSGenValueRadiusFractionBuilder.h"
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

template<> KSGenPositionCylindricalCompositeBuilder::~KComplexElement() = default;

STATICINT sKSGenPositionCylindricalCompositeStructure =
    //! the name of this position creator.
    KSGenPositionCylindricalCompositeBuilder::Attribute<std::string>("name") +
    //! the name of a surface in kgeobag to use as the origin of coordinates.
    KSGenPositionCylindricalCompositeBuilder::Attribute<std::string>("surface") +
    //! the name of a space in kgeobag to use as the origin of coordinates.
    KSGenPositionCylindricalCompositeBuilder::Attribute<std::string>("space") +
    KSGenPositionCylindricalCompositeBuilder::Attribute<std::string>("r") +
    KSGenPositionCylindricalCompositeBuilder::Attribute<std::string>("phi") +
    KSGenPositionCylindricalCompositeBuilder::Attribute<std::string>("z") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueFix>("r_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueSet>("r_set") +
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueList>("r_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueUniform>("r_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueGauss>("r_gauss") +
    //! creates a radius cylindrical value to set the r coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueRadiusCylindrical>("r_cylindrical") +
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueRadiusFraction>("r_fraction") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueFix>("phi_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueSet>("phi_set") +
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueList>("phi_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueUniform>("phi_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueGauss>("phi_gauss") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueFix>("z_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueSet>("z_set") +
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueList>("z_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueUniform>("z_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueGauss>("z_gauss");

STATICINT sKSGenPositionCylindricalComposite =
    KSRootBuilder::ComplexElement<KSGenPositionCylindricalComposite>("ksgen_position_cylindrical_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenPositionCylindricalCompositeStructureROOT =
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueFormula>("r_formula") +
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueHistogram>("r_histogram") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueFormula>("phi_formula") +
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueHistogram>("phi_histogram") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|phi|z) coordinate.
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueFormula>("z_formula") +
    KSGenPositionCylindricalCompositeBuilder::ComplexElement<KSGenValueHistogram>("z_histogram");
#endif

}  // namespace katrin
