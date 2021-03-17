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
    for (auto& calculator : fCalculators) {
        delete calculator;
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
    for (auto& calculator : fCalculators) {
        aScattering->AddCalculator(calculator);
    }
    fCalculators.clear();
    return;
}

template<> KSIntCalculatorKESSSetBuilder::~KComplexElement() = default;

static int __attribute__((__unused__)) sKSIntCalculatorKESSStructure =
    KSIntCalculatorKESSSetBuilder::Attribute<std::string>("name") +
    KSIntCalculatorKESSSetBuilder::Attribute<bool>("elastic") +
    KSIntCalculatorKESSSetBuilder::Attribute<std::string>("inelastic") +
    KSIntCalculatorKESSSetBuilder::Attribute<bool>("photo_absorbtion") +
    KSIntCalculatorKESSSetBuilder::Attribute<std::string>("auger_relaxation");

}  // namespace katrin
