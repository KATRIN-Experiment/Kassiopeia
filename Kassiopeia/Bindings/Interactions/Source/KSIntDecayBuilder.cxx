#include "KSIntDecayBuilder.h"

#include "KSIntDecayCalculatorDeathConstRateBuilder.h"
#include "KSIntDecayCalculatorFerencBBRTransitionBuilder.h"
#include "KSIntDecayCalculatorFerencIonisationBuilder.h"
#include "KSIntDecayCalculatorFerencSpontaneousBuilder.h"
#include "KSIntDecayCalculatorGlukhovDeExcitationBuilder.h"
#include "KSIntDecayCalculatorGlukhovExcitationBuilder.h"
#include "KSIntDecayCalculatorGlukhovIonisationBuilder.h"
#include "KSIntDecayCalculatorGlukhovSpontaneousBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

KSIntDecayCalculatorSet::KSIntDecayCalculatorSet() = default;
KSIntDecayCalculatorSet::~KSIntDecayCalculatorSet() = default;

template<> KSIntDecayBuilder::~KComplexElement() = default;

STATICINT sKSIntDecayStructure =
    KSIntDecayBuilder::Attribute<std::string>("name") + KSIntDecayBuilder::Attribute<std::string>("calculator") +
    KSIntDecayBuilder::Attribute<std::string>("calculators") + KSIntDecayBuilder::Attribute<double>("enhancement") +
    KSIntDecayBuilder::ComplexElement<KSIntDecayCalculatorDeathConstRate>("decay_death_const_rate") +
    KSIntDecayBuilder::ComplexElement<KSIntDecayCalculatorGlukhovSpontaneous>("decay_glukhov_spontaneous") +
    KSIntDecayBuilder::ComplexElement<KSIntDecayCalculatorGlukhovIonisation>("decay_glukhov_ionisation") +
    KSIntDecayBuilder::ComplexElement<KSIntDecayCalculatorGlukhovExcitation>("decay_glukhov_excitation") +
    KSIntDecayBuilder::ComplexElement<KSIntDecayCalculatorGlukhovDeExcitation>("decay_glukhov_deexcitation") +
    KSIntDecayBuilder::ComplexElement<KSIntDecayCalculatorFerencBBRTransition>("decay_ferenc_bbr") +
    KSIntDecayBuilder::ComplexElement<KSIntDecayCalculatorFerencSpontaneous>("decay_ferenc_spontaneous") +
    KSIntDecayBuilder::ComplexElement<KSIntDecayCalculatorFerencIonisation>("decay_ferenc_ionisation");

STATICINT sKSIntDecay = KSRootBuilder::ComplexElement<KSIntDecay>("ksint_decay");

}  // namespace katrin
