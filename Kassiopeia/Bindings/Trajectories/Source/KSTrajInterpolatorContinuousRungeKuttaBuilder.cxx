#include "KSTrajInterpolatorContinuousRungeKuttaBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSTrajInterpolatorContinuousRungeKuttaBuilder::~KComplexElement() = default;

STATICINT sKSTrajInterpolatorContinuousRungeKuttaStructure =
    KSTrajInterpolatorContinuousRungeKuttaBuilder::Attribute<std::string>("name");

STATICINT sToolboxKSTrajInterpolatorContinuousRungeKutta =
    KSRootBuilder::ComplexElement<KSTrajInterpolatorContinuousRungeKutta>("kstraj_interpolator_crk");

}  // namespace katrin
