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

template<> KSGenPositionSphericalCompositeBuilder::~KComplexElement() {}

STATICINT sKSGenPositionSphericalCompositeStructure =
    KSGenPositionSphericalCompositeBuilder::Attribute<string>("name") +
    KSGenPositionSphericalCompositeBuilder::Attribute<string>("surface") +
    KSGenPositionSphericalCompositeBuilder::Attribute<string>("space") +
    KSGenPositionSphericalCompositeBuilder::Attribute<string>("r") +
    KSGenPositionSphericalCompositeBuilder::Attribute<string>("theta") +
    KSGenPositionSphericalCompositeBuilder::Attribute<string>("phi") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFix>("r_fix") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueSet>("r_set") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueList>("r_list") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueUniform>("r_uniform") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueGauss>("r_gauss") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueRadiusSpherical>("r_spherical") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFix>("theta_fix") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueSet>("theta_set") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueList>("theta_list") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueUniform>("theta_uniform") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueGauss>("theta_gauss") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueAngleSpherical>("theta_spherical") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFix>("phi_fix") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueSet>("phi_set") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueList>("phi_list") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueUniform>("phi_uniform") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueGauss>("phi_gauss");

STATICINT sKSGenPositionSphericalComposite =
    KSRootBuilder::ComplexElement<KSGenPositionSphericalComposite>("ksgen_position_spherical_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenPositionSphericalCompositeStructureROOT =
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFormula>("r_formula") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueHistogram>("r_histogram") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFormula>("theta_formula") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueHistogram>("theta_histogram") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueFormula>("phi_formula") +
    KSGenPositionSphericalCompositeBuilder::ComplexElement<KSGenValueHistogram>("phi_histogram");
#endif

}  // namespace katrin
