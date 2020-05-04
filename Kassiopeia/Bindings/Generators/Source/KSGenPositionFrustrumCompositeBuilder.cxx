#include "KSGenPositionFrustrumCompositeBuilder.h"

#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueRadiusCylindricalBuilder.h"
#include "KSGenValueRadiusFractionBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSGenValueZFrustrumBuilder.h"
#include "KSRootBuilder.h"

#ifdef Kassiopeia_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#include "KSGenValueHistogramBuilder.h"
#endif

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenPositionFrustrumCompositeBuilder::~KComplexElement() {}

STATICINT sKSGenPositionFrustrumCompositeStructure =
    KSGenPositionFrustrumCompositeBuilder::Attribute<string>("name") +
    KSGenPositionFrustrumCompositeBuilder::Attribute<string>("surface") +
    KSGenPositionFrustrumCompositeBuilder::Attribute<string>("space") +
    KSGenPositionFrustrumCompositeBuilder::Attribute<string>("r") +
    KSGenPositionFrustrumCompositeBuilder::Attribute<string>("phi") +
    KSGenPositionFrustrumCompositeBuilder::Attribute<string>("z") +
    KSGenPositionFrustrumCompositeBuilder::Attribute<string>("r1") +
    KSGenPositionFrustrumCompositeBuilder::Attribute<string>("r2") +
    KSGenPositionFrustrumCompositeBuilder::Attribute<string>("z1") +
    KSGenPositionFrustrumCompositeBuilder::Attribute<string>("z2") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueFix>("r_fix") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueFix>("r1_fix") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueFix>("r2_fix") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueSet>("r_set") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueList>("r_list") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueUniform>("r_uniform") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueGauss>("r_gauss") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueRadiusCylindrical>("r_cylindrical") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueRadiusFraction>("r_fraction") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueFix>("phi_fix") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueSet>("phi_set") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueList>("phi_list") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueUniform>("phi_uniform") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueGauss>("phi_gauss") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueFix>("z_fix") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueFix>("z1_fix") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueFix>("z2_fix") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueSet>("z_set") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueList>("z_list") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueUniform>("z_uniform") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueZFrustrum>("z_frustrum") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueGauss>("z_gauss");

STATICINT sKSGenPositionFrustrumComposite =
    KSRootBuilder::ComplexElement<KSGenPositionFrustrumComposite>("ksgen_position_frustrum_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenPositionFrustrumCompositeStructureROOT =
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueFormula>("r_formula") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueHistogram>("r_histogram") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueFormula>("phi_formula") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueHistogram>("phi_histogram") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueFormula>("z_formula") +
    KSGenPositionFrustrumCompositeBuilder::ComplexElement<KSGenValueHistogram>("z_histogram");
#endif

}  // namespace katrin
