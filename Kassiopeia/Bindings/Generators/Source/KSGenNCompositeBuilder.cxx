#include "KSGenNCompositeBuilder.h"

#include "KSGenValueFixBuilder.h"
#include "KSGenValueGaussBuilder.h"
#include "KSGenValueListBuilder.h"
#include "KSGenValueParetoBuilder.h"
#include "KSGenValueSetBuilder.h"
#include "KSGenValueUniformBuilder.h"
#include "KSRootBuilder.h"

#ifdef Kassiopeia_USE_ROOT
#include "KSGenValueFormulaBuilder.h"
#endif

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSGenNCompositeBuilder::~KComplexElement() = default;

STATICINT sKSGenNCompositeStructure = KSGenNCompositeBuilder::Attribute<std::string>("name") +
                                      KSGenNCompositeBuilder::Attribute<std::string>("n_value") +
                                      KSGenNCompositeBuilder::ComplexElement<KSGenValueFix>("n_fix") +
                                      KSGenNCompositeBuilder::ComplexElement<KSGenValueSet>("n_set") +
                                      KSGenNCompositeBuilder::ComplexElement<KSGenValueList>("n_list") +
                                      KSGenNCompositeBuilder::ComplexElement<KSGenValueUniform>("n_uniform") +
                                      KSGenNCompositeBuilder::ComplexElement<KSGenValueGauss>("n_gauss") +
                                      KSGenNCompositeBuilder::ComplexElement<KSGenValuePareto>("n_pareto");

STATICINT sKSGenNComposite = KSRootBuilder::ComplexElement<KSGenNComposite>("ksgen_n_composite");

#ifdef Kassiopeia_USE_ROOT
STATICINT sKSGenNCompositeStructureROOT = KSGenNCompositeBuilder::ComplexElement<KSGenValueFormula>("n_formula");
#endif

}  // namespace katrin
