#include "KSIntDecayCalculatorDeathConstRateBuilder.h"

#include "KSGenGeneratorCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntDecayCalculatorDeathConstRateBuilder::~KComplexElement() = default;

STATICINT sKSIntDecayCalculatorDeathConstRateBuilderStructure =
    KSIntDecayCalculatorDeathConstRateBuilder::Attribute<std::string>("name") +
    KSIntDecayCalculatorDeathConstRateBuilder::Attribute<double>("life_time") +
    KSIntDecayCalculatorDeathConstRateBuilder::Attribute<long long>("target_pid") +
    KSIntDecayCalculatorDeathConstRateBuilder::Attribute<long long>("min_pid") +
    KSIntDecayCalculatorDeathConstRateBuilder::Attribute<long long>("max_pid") +
    KSIntDecayCalculatorDeathConstRateBuilder::ComplexElement<KSGenGeneratorComposite>("decay_product_generator");

STATICINT sToolboxKSIntDecayCalculatorDeathConstRate =
    KSRootBuilder::ComplexElement<KSIntDecayCalculatorDeathConstRate>("ksint_decay_calculator_death_const_rate");
}  // namespace katrin
