//
// Created by trost on 27.05.15.
//

#include "KSIntDecayCalculatorFerencSpontaneousBuilder.h"

#include "KSGenGeneratorCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntDecayCalculatorFerencSpontaneousBuilder::~KComplexElement() {}

STATICINT sKSIntDecayCalculatorFerencSpontaneousBuilderStructure =
    KSIntDecayCalculatorFerencSpontaneousBuilder::Attribute<string>("name") +
    KSIntDecayCalculatorFerencSpontaneousBuilder::Attribute<long long>("target_pid") +
    KSIntDecayCalculatorFerencSpontaneousBuilder::Attribute<long long>("min_pid") +
    KSIntDecayCalculatorFerencSpontaneousBuilder::Attribute<long long>("max_pid") +
    KSIntDecayCalculatorFerencSpontaneousBuilder::ComplexElement<KSGenGeneratorComposite>("decay_product_generator");

STATICINT sToolboxKSIntDecayCalculatorFerencSpontaneous =
    KSRootBuilder::ComplexElement<KSIntDecayCalculatorFerencSpontaneous>("ksint_decay_calculator_ferenc_spontaneous");

}  // namespace katrin
