#include "KSIntDecayCalculatorGlukhovDeExcitationBuilder.h"

#include "KSGenGeneratorCompositeBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

template<> KSIntDecayCalculatorGlukhovDeExcitationBuilder::~KComplexElement() {}

STATICINT sKSIntDecayCalculatorGlukhovDeExcitationBuilderStructure =
    KSIntDecayCalculatorGlukhovDeExcitationBuilder::Attribute<string>("name") +
    KSIntDecayCalculatorGlukhovDeExcitationBuilder::Attribute<long long>("target_pid") +
    KSIntDecayCalculatorGlukhovDeExcitationBuilder::Attribute<long long>("min_pid") +
    KSIntDecayCalculatorGlukhovDeExcitationBuilder::Attribute<long long>("max_pid") +
    KSIntDecayCalculatorGlukhovDeExcitationBuilder::Attribute<double>("temperature") +
    KSIntDecayCalculatorGlukhovDeExcitationBuilder::ComplexElement<KSGenGeneratorComposite>("decay_product_generator");

STATICINT sToolboxKSIntDecayCalculatorGlukhovDeExcitation =
    KSRootBuilder::ComplexElement<KSIntDecayCalculatorGlukhovDeExcitation>(
        "ksint_decay_calculator_glukhov_deexcitation");
}  // namespace katrin
