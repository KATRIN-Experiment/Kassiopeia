#include "KSIntScatteringBuilder.h"

#include "KSIntCalculatorArgonBuilder.h"
#include "KSIntCalculatorConstantBuilder.h"
#include "KSIntCalculatorHydrogenBuilder.h"
#include "KSIntCalculatorIonBuilder.h"
#include "KSIntCalculatorKESSBuilder.h"
#include "KSIntCalculatorMottBuilder.h"
#include "KSIntDensityConstantBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

KSIntCalculatorSet::KSIntCalculatorSet() = default;
KSIntCalculatorSet::~KSIntCalculatorSet() = default;

template<> KSIntScatteringBuilder::~KComplexElement() = default;

STATICINT sKSIntScattering = KSRootBuilder::ComplexElement<KSIntScattering>("ksint_scattering");

STATICINT sKSIntScatteringStructure =
    //! the name of the interaction.
    KSIntScatteringBuilder::Attribute<std::string>("name") +
    //! if true, the interaction will split tracks whenever a scattering occurs.
    KSIntScatteringBuilder::Attribute<bool>("split") +
    //! the name of a density which has been defined earlier in the configuration.
    KSIntScatteringBuilder::Attribute<std::string>("density") +
    //! the name of a calculator which has been defined earlier in the configuration.
    KSIntScatteringBuilder::Attribute<std::string>("calculator") +
    //! the name of a tag containing calculators which has been defined earlier in the configuration.
    KSIntScatteringBuilder::Attribute<std::string>("calculators") +
    KSIntScatteringBuilder::Attribute<double>("enhancement") +
    //! builds a constant density.
    KSIntScatteringBuilder::ComplexElement<KSIntDensityConstant>("density_constant") +
    //! builds a constant calculator.
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorConstant>("calculator_constant") +
    //! builds a set of hydrogen calculators.
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorHydrogenSet>("calculator_hydrogen") +
    //! builds an ion calculator, used for ionization of gas by ions.
    //!
    //! This affects the following ions: H^+, H_2^+, H_3^+, H^-, D^+, D_2^+, D_3^+, D^-, T^+, T_2^+, T_3^+, T^-.
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorIon>("calculator_ion") +
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorArgonSet>("calculator_argon") +
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorKESSSet>("calculator_kess") +
    //! builds a mott calculator
    KSIntScatteringBuilder::ComplexElement<KSIntCalculatorMott>("calculator_mott");
}  // namespace katrin
