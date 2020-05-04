#include "KSIntCalculatorKESSBuilder.h"

#include "KSRootBuilder.h"

using namespace Kassiopeia;
using namespace std;

namespace katrin
{

KSIntCalculatorKESSSet::KSIntCalculatorKESSSet() :
    fName("anonymous"),
    fElastic(true),
    fInelastic("bethe_fano"),
    fPhotoAbsorbtion(true),
    fAugerRelaxation(false)
{}
KSIntCalculatorKESSSet::~KSIntCalculatorKESSSet()
{
    for (auto tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        delete (*tIt);
    }
}


void KSIntCalculatorKESSSet::AddCalculator(KSIntCalculator* aCalculator)
{
    KToolbox::GetInstance().Add(aCalculator);
    fCalculators.push_back(aCalculator);
    return;
}

void KSIntCalculatorKESSSet::ReleaseCalculators(KSIntScattering* aScattering)
{
    for (auto tIt = fCalculators.begin(); tIt != fCalculators.end(); tIt++) {
        aScattering->AddCalculator(*tIt);
    }
    fCalculators.clear();
    return;
}

template<> KSIntCalculatorKESSSetBuilder::~KComplexElement() {}

static int __attribute__((__unused__)) sKSIntCalculatorKESSStructure =
    KSIntCalculatorKESSSetBuilder::Attribute<string>("name") +
    KSIntCalculatorKESSSetBuilder::Attribute<bool>("elastic") +
    KSIntCalculatorKESSSetBuilder::Attribute<string>("inelastic") +
    KSIntCalculatorKESSSetBuilder::Attribute<bool>("photo_absorbtion") +
    KSIntCalculatorKESSSetBuilder::Attribute<string>("auger_relaxation");

}  // namespace katrin
