#include "KSGenSpinCompositeBuilder.h"

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

template<> KSGenSpinCompositeBuilder::~KComplexElement() {}

STATICINT sKSGenSpinCompositeStructure =
    KSGenSpinCompositeBuilder::Attribute<string>("name") + KSGenSpinCompositeBuilder::Attribute<string>("theta") +
    KSGenSpinCompositeBuilder::Attribute<string>("phi") + KSGenSpinCompositeBuilder::Attribute<string>("surface") +
    KSGenSpinCompositeBuilder::Attribute<string>("space") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueFix>("theta_fix") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueSet>("theta_set") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueList>("theta_list") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueUniform>("theta_uniform") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueGauss>("theta_gauss") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueGeneralizedGauss>("theta_generalized_gauss") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueAngleCosine>("theta_cosine") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueAngleSpherical>("theta_spherical") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueFix>("phi_fix") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueSet>("phi_set") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueList>("phi_list") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueUniform>("phi_uniform") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueGauss>("phi_gauss") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueGeneralizedGauss>("phi_generalized_gauss");

STATICINT sKSGenSpinComposite = KSRootBuilder::ComplexElement<KSGenSpinComposite>("ksgen_spin_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenSpinCompositeStructureROOT =
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueFormula>("theta_formula") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueHistogram>("theta_histogram") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueFormula>("phi_formula") +
    KSGenSpinCompositeBuilder::ComplexElement<KSGenValueHistogram>("phi_histogram");
#endif

}  // namespace katrin
