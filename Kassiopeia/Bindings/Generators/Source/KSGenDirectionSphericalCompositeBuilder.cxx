#include "KSGenDirectionSphericalCompositeBuilder.h"

#include "KSGenValueAngleCosineBuilder.h"
#include "KSGenValueAngleSphericalBuilder.h"
#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueGeneralizedGaussBuilder.h"
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
    KSGenDirectionSphericalCompositeBuilder::Attribute<std::string>("name") +
    KSGenDirectionSphericalCompositeBuilder::Attribute<std::string>("theta") +
    KSGenDirectionSphericalCompositeBuilder::Attribute<std::string>("phi") +
    KSGenDirectionSphericalCompositeBuilder::Attribute<std::string>("surface") +
    KSGenDirectionSphericalCompositeBuilder::Attribute<std::string>("space") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueFix>("theta_fix") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueSet>("theta_set") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueList>("theta_list") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueUniform>("theta_uniform") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueGauss>("theta_gauss") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueGeneralizedGauss>("theta_generalized_gauss") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueAngleCosine>("theta_cosine") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueAngleSpherical>("theta_spherical") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueFix>("phi_fix") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueSet>("phi_set") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueList>("phi_list") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueUniform>("phi_uniform") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueGauss>("phi_gauss") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueGeneralizedGauss>("phi_generalized_gauss");

STATICINT sKSGenDirectionSphericalComposite =
    KSRootBuilder::ComplexElement<KSGenDirectionSphericalComposite>("ksgen_direction_spherical_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenDirectionSphericalCompositeStructureROOT =
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueFormula>("theta_formula") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueHistogram>("theta_histogram") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueFormula>("phi_formula") +
    KSGenDirectionSphericalCompositeBuilder::ComplexElement<KSGenValueHistogram>("phi_histogram");
#endif

}  // namespace katrin
