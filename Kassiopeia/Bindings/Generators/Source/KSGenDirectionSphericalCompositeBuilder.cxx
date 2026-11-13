#include "KSGenDirectionSphericalCompositeBuilder.h"

#include "KSGenValueAngleCosineBuilder.h"
#include "KSGenValueAngleSphericalBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueGeneralizedGaussBuilder.h"
#include "KSGenValueFermiBuilder.h"
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

template<> KSGenDirectionSphericalCompositeBuilder::~KComplexElement() = default;

STATICINT sKSGenDirectionSphericalCompositeStructure =
    //! the name of this direction creator.
    KSGenDirectionSphericalCompositeBuilder::Attribute<std::string>("name") +
    KSGenDirectionSphericalCompositeBuilder::Attribute<std::string>("theta") +
    KSGenDirectionSphericalCompositeBuilder::Attribute<std::string>("phi") +
    //! the name of a surface in kgeobag to use as the origin of coordinates.
    KSGenDirectionSphericalCompositeBuilder::Attribute<std::string>("surface") +
    //! the name of a space in kgeobag to use as the origin of coordinates.
    KSGenDirectionSphericalCompositeBuilder::Attribute<std::string>("space") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (theta|phi) coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueFix>("theta_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (theta|phi) coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueSet>("theta_set") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueList>("theta_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (theta|phi) coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueUniform>("theta_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (theta|phi) coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueGauss>("theta_gauss") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueGeneralizedGauss>("theta_generalized_gauss") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueFermi>("theta_fermi") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueAngleCosine>("theta_cosine") +
    //! creates an angle spherical value to set the theta coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueAngleSpherical>("theta_spherical") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (theta|phi) coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueFix>("phi_fix") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (theta|phi) coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueSet>("phi_set") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueList>("phi_list") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (theta|phi) coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueUniform>("phi_uniform") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (theta|phi) coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueGauss>("phi_gauss") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueGeneralizedGauss>("phi_generalized_gauss") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueFermi>("phi_fermi");

STATICINT sKSGenDirectionSphericalComposite =
    KSRootBuilder::ComplexElement<KSGenDirectionSphericalComposite>("ksgen_direction_spherical_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenDirectionSphericalCompositeStructureROOT =
    //! creates a (fix|set|uniform|gauss|formula) value to set the (theta|phi) coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueFormula>("theta_formula") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueHistogram>("theta_histogram") +
    //! creates a (fix|set|uniform|gauss|formula) value to set the (theta|phi) coordinate.
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueFormula>("phi_formula") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueHistogram>("phi_histogram");
#endif

}  // namespace katrin
