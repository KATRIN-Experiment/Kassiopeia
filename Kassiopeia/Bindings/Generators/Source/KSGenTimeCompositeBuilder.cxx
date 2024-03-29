#include "KSGenTimeCompositeBuilder.h"

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

template<> KSGenTimeCompositeBuilder::~KComplexElement() = default;

STATICINT sKSGenTimeCompositeStructure =
    KSGenTimeCompositeBuilder::Attribute<std::string>("name") +
    KSGenTimeCompositeBuilder::Attribute<std::string>("time_value") +
    KSGenTimeCompositeBuilder::ComplexElement<KSGenValueFix>("time_fix") +
    KSGenTimeCompositeBuilder::ComplexElement<KSGenValueSet>("time_set") +
    KSGenTimeCompositeBuilder::ComplexElement<KSGenValueList>("time_list") +
    KSGenTimeCompositeBuilder::ComplexElement<KSGenValueUniform>("time_uniform") +
    KSGenTimeCompositeBuilder::ComplexElement<KSGenValueGauss>("time_gauss") +
    KSGenTimeCompositeBuilder::ComplexElement<KSGenValueGeneralizedGauss>("time_generalized_gauss") +
    KSGenTimeCompositeBuilder::ComplexElement<KSGenValueFermi>("time_fermi");

STATICINT sKSGenTimeComposite = KSRootBuilder::ComplexElement<KSGenTimeComposite>("ksgen_time_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenTimeCompositeStructureROOT =
    KSGenTimeCompositeBuilder::ComplexElement<KSGenValueFormula>("time_formula") +
    KSGenTimeCompositeBuilder::ComplexElement<KSGenValueHistogram>("time_histogram");
#endif

}  // namespace katrin
