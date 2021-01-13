//
// Created by trost on 27.05.15.
//

#include "KSIntDecayCalculatorFerencBBRTransitionBuilder.h"

#include "KSGenGeneratorCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntDecayCalculatorFerencBBRTransitionBuilder::~KComplexElement() = default;

STATICINT sKSIntDecayCalculatorFerencBBRTransitionBuilderStructure =
    KSIntDecayCalculatorFerencBBRTransitionBuilder::Attribute<std::string>("name") +
    KSIntDecayCalculatorFerencBBRTransitionBuilder::Attribute<long long>("target_pid") +
    KSIntDecayCalculatorFerencBBRTransitionBuilder::Attribute<long long>("min_pid") +
    KSIntDecayCalculatorFerencBBRTransitionBuilder::Attribute<long long>("max_pid") +
    KSIntDecayCalculatorFerencBBRTransitionBuilder::Attribute<double>("temperature") +
    KSIntDecayCalculatorFerencBBRTransitionBuilder::ComplexElement<KSGenGeneratorComposite>("decay_product_generator");

STATICINT sToolboxKSIntDecayCalculatorFerencBBRTransition =
    KSRootBuilder::ComplexElement<KSIntDecayCalculatorFerencBBRTransition>(
        "ksint_decay_calculator_ferenc_bbr_transition");
}  // namespace katrin
