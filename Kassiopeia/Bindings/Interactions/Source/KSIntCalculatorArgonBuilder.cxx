#include "KSIntCalculatorArgonBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

KSIntCalculatorArgonSet::KSIntCalculatorArgonSet() :
    fName("anonymous"),
    fSingleIonisation(true),
    fDoubleIonisation(true),
    fExcitation(true),
    fElastic(true)
{}

KSIntCalculatorArgonSet::~KSIntCalculatorArgonSet()
{
    for (auto& calculator : fCalculators) {
        delete calculator;
    }
}

void KSIntCalculatorArgonSet::AddCalculator(KSIntCalculator* aCalculator)
{
    katrin::KToolbox::GetInstance().Add(aCalculator);
    fCalculators.push_back(aCalculator);
    return;
}

void KSIntCalculatorArgonSet::ReleaseCalculators(KSIntScattering* aScattering)
{
    for (auto& calculator : fCalculators) {
        aScattering->AddCalculator(calculator);
    }
    fCalculators.clear();
    return;
}

template<> KSIntCalculatorArgonSetBuilder::~KComplexElement() = default;

STATICINT sKSIntCalculatorArgonStructure = KSIntCalculatorArgonSetBuilder::Attribute<std::string>("name") +
                                           KSIntCalculatorArgonSetBuilder::Attribute<bool>("elastic") +
                                           KSIntCalculatorArgonSetBuilder::Attribute<bool>("excitation") +
                                           KSIntCalculatorArgonSetBuilder::Attribute<bool>("single_ionisation") +
                                           KSIntCalculatorArgonSetBuilder::Attribute<bool>("double_ionisation");
}  // namespace katrin
