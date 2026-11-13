#include "KSGenPositionSphericalCompositeBuilder.h"

#include "KSGenValueAngleSphericalBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueRadiusSphericalBuilder.h"
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

template<> KSGenPositionSphericalCompositeBuilder::~KComplexElement() = default;

STATICINT sKSGenPositionSphericalCompositeStructure =
    //! the name of this position creator.
    KSGenPositionSphericalCompositeBuilder::Attribute<std::string>("name") +
    //! the name of a surface in kgeobag to use as the origin of coordinates.
    KSGenPositionSphericalCompositeBuilder::Attribute<std::string>("surface") +
    //! the name of a space in kgeobag to use as the origin of coordinates.
    KSGenPositionSphericalCompositeBuilder::Attribute<std::string>("space") +
    KSGenPositionSphericalCompositeBuilder::Attribute<std::string>("r") +
    KSGenPositionSphericalCompositeBuilder::Attribute<std::string>("theta") +
    KSGenPositionSphericalCompositeBuilder::Attribute<std::string>("phi") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFix>("r_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueSet>("r_set") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueList>("r_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueUniform>("r_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueGauss>("r_gauss") +
    //! creates a radius spherical value to set the r coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueRadiusSpherical>("r_spherical") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFix>("theta_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueSet>("theta_set") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueList>("theta_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueUniform>("theta_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueGauss>("theta_gauss") +
    //! creates an angle spherical value to set the theta coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueAngleSpherical>("theta_spherical") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFix>("phi_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueSet>("phi_set") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueList>("phi_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueUniform>("phi_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueGauss>("phi_gauss");

STATICINT sKSGenPositionSphericalComposite =
    KSRootBuilder::ComplexElement<KSGenPositionSphericalComposite>("ksgen_position_spherical_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenPositionSphericalCompositeStructureROOT =
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFormula>("r_formula") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueHistogram>("r_histogram") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFormula>("theta_formula") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueHistogram>("theta_histogram") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (r|theta|phi) coordinate.
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFormula>("phi_formula") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueHistogram>("phi_histogram");
#endif

}  // namespace katrin
