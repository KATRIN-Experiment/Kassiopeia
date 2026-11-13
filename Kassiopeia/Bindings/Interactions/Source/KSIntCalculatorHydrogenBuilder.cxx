#include "KSIntCalculatorHydrogenBuilder.h"

#include "KSIntCalculatorConstantBuilder.h"
#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

KSIntCalculatorHydrogenSet::KSIntCalculatorHydrogenSet() :
    fName("anonymous"),
    fElastic(true),
    fExcitation(true),
    fIonisation(true),
    fMolecule("hydrogen")
{}
KSIntCalculatorHydrogenSet::~KSIntCalculatorHydrogenSet()
{
    for (auto& calculator : fCalculators) {
        delete calculator;
    }
}


void KSIntCalculatorHydrogenSet::AddCalculator(KSIntCalculator* aCalculator)
{
    katrin::KToolbox::GetInstance().Add(aCalculator);
    fCalculators.push_back(aCalculator);
    return;
}

void KSIntCalculatorHydrogenSet::ReleaseCalculators(KSIntScattering* aScattering)
{
    for (auto& calculator : fCalculators) {
        aScattering->AddCalculator(calculator);
    }
    fCalculators.clear();
    return;
}

template<> KSIntCalculatorHydrogenSetBuilder::~KComplexElement() = default;

STATICINT sKSIntCalculatorHydrogenStructure = KSIntCalculatorHydrogenSetBuilder::Attribute<std::string>("name") +
                                              //! if true, all elastic processes are enabled.
                                              KSIntCalculatorHydrogenSetBuilder::Attribute<bool>("elastic") +
                                              //! if true, all excitation processes are enabled.
                                              KSIntCalculatorHydrogenSetBuilder::Attribute<bool>("excitation") +
                                              //! if true, all ionisation processes are enabled. note british spelling!
                                              KSIntCalculatorHydrogenSetBuilder::Attribute<bool>("ionisation") +
                                              //! if set to tritium, the rot20 calculator is also added
                                              KSIntCalculatorHydrogenSetBuilder::Attribute<std::string>("molecule");

}  // namespace katrin
