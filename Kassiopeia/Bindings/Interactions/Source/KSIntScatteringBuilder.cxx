#include "KSIntScatteringBuilder.h"

#include "KSIntCalculatorArgonBuilder.h"
#include "KSIntCalculatorConstantBuilder.h"
#include "KSIntCalculatorHydrogenBuilder.h"
#include "KSIntCalculatorIonBuilder.h"
#include "KSIntCalculatorKESSBuilder.h"
#include "KSIntDensityConstantBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

KSIntCalculatorSet::KSIntCalculatorSet() {}
KSIntCalculatorSet::~KSIntCalculatorSet() {}

template<> KSIntScatteringBuilder::~KComplexElement() {}

STATICINT sKSIntScattering = KSRootBuilder::ComplexElement<KSIntScattering>("ksint_scattering");

STATICINT sKSIntScatteringStructure =
    KSIntScatteringBuilder::Attribute<string>("name") + KSIntScatteringBuilder::Attribute<bool>("split") +
    KSIntScatteringBuilder::Attribute<string>("density") + KSIntScatteringBuilder::Attribute<string>("calculator") +
    KSIntScatteringBuilder::Attribute<string>("calculators") +
    KSIntScatteringBuilder::Attribute<double>("enhancement") +
    KSIntScatteringBuilder::ComplexElement<KSIntDensityConstant>("density_constant") +
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorConstant>("calculator_constant") +
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorHydrogenSet>("calculator_hydrogen") +
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorIon>("calculator_ion") +
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorArgonSet>("calculator_argon") +
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorKESSSet>("calculator_kess");
}  // namespace katrin
