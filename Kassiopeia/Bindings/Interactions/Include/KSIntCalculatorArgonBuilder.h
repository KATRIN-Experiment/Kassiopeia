#ifndef Kassiopeia_KSIntCalculatorArgonBuilder_h_
#define Kassiopeia_KSIntCalculatorArgonBuilder_h_

#include "KComplexElement.hh"
#include "KField.h"
#include "KSIntCalculatorArgon.h"
#include "KSIntScatteringBuilder.h"

using namespace Kassiopeia;
namespace katrin
{

class KSIntCalculatorArgonSet : public KSIntCalculatorSet
{
  public:
    KSIntCalculatorArgonSet();
    ~KSIntCalculatorArgonSet() override;

  public:
    void AddCalculator(KSIntCalculator* aCalculator) override;
    void ReleaseCalculators(KSIntScattering* aScattering) override;

  private:
    K_SET_GET(string, Name)
    K_SET_GET(bool, SingleIonisation)
    K_SET_GET(bool, DoubleIonisation)
    K_SET_GET(bool, Excitation)
    K_SET_GET(bool, Elastic)
    std::vector<KSIntCalculator*> fCalculators;
};


typedef KComplexElement<KSIntCalculatorArgonSet> KSIntCalculatorArgonSetBuilder;

template<> inline bool KSIntCalculatorArgonSetBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KSIntCalculatorArgonSet::SetName);
        return true;
    }
    else if (aContainer->GetName() == "elastic") {
        aContainer->CopyTo(fObject, &KSIntCalculatorArgonSet::SetElastic);
        return true;
    }
    else if (aContainer->GetName() == "excitation") {
        aContainer->CopyTo(fObject, &KSIntCalculatorArgonSet::SetExcitation);
        return true;
    }
    else if (aContainer->GetName() == "single_ionisation") {
        aContainer->CopyTo(fObject, &KSIntCalculatorArgonSet::SetSingleIonisation);
        return true;
    }
    else if (aContainer->GetName() == "double_ionisation") {
        aContainer->CopyTo(fObject, &KSIntCalculatorArgonSet::SetDoubleIonisation);
        return true;
    }

    return false;
}

template<> inline bool KSIntCalculatorArgonSetBuilder::End()
{
    if (fObject->GetElastic()) {
        KSIntCalculator* aIntCalculator;

        aIntCalculator = new KSIntCalculatorArgonElastic();
        aIntCalculator->SetName(fObject->GetName() + "_elastic");
        aIntCalculator->SetTag(fObject->GetName());
        fObject->AddCalculator(aIntCalculator);
    }

    if (fObject->GetSingleIonisation()) {
        KSIntCalculator* aIntCalculator;

        aIntCalculator = new KSIntCalculatorArgonSingleIonisation();
        aIntCalculator->SetName(fObject->GetName() + "_single_ionisation");
        aIntCalculator->SetTag(fObject->GetName());
        fObject->AddCalculator(aIntCalculator);
    }

    if (fObject->GetDoubleIonisation()) {
        KSIntCalculator* aIntCalculator;

        aIntCalculator = new KSIntCalculatorArgonDoubleIonisation();
        aIntCalculator->SetName(fObject->GetName() + "_double_ionisation");
        aIntCalculator->SetTag(fObject->GetName());
        fObject->AddCalculator(aIntCalculator);
    }

    if (fObject->GetExcitation()) {
        KSIntCalculator* aIntCalculator;

        for (unsigned int i = 0; i < 25; ++i) {
            std::stringstream tmp;
            tmp << (i + 1);
            aIntCalculator = new KSIntCalculatorArgonExcitation();
            aIntCalculator->SetName(fObject->GetName() + "_excitation_state_" + tmp.str());
            aIntCalculator->SetTag(fObject->GetName());
            static_cast<KSIntCalculatorArgonExcitation*>(aIntCalculator)->SetExcitationState(i + 1);
            fObject->AddCalculator(aIntCalculator);
        }
    }
    return true;
}
}  // namespace katrin

#endif
