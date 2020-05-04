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
    for (auto tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        delete (*tIt);
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
    for (auto tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        aScattering->AddCalculator(*tIt);
    }
    fCalculators.clear();
    return;
}

template<> KSIntCalculatorHydrogenSetBuilder::~KComplexElement() {}

STATICINT sKSIntCalculatorHydrogenStructure = KSIntCalculatorHydrogenSetBuilder::Attribute<string>("name") +
                                              KSIntCalculatorHydrogenSetBuilder::Attribute<bool>("elastic") +
                                              KSIntCalculatorHydrogenSetBuilder::Attribute<bool>("excitation") +
                                              KSIntCalculatorHydrogenSetBuilder::Attribute<bool>("ionisation") +
                                              KSIntCalculatorHydrogenSetBuilder::Attribute<string>("molecule");

}  // namespace katrin
