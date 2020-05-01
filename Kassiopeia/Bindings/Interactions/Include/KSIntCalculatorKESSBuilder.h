#ifndef Kassiopeia_KSIntCalculatorKESSBuilder_h_
#define Kassiopeia_KSIntCalculatorKESSBuilder_h_

#include "KComplexElement.hh"
#include "KESSElasticElsepa.h"
#include "KESSInelasticBetheFano.h"
#include "KESSInelasticPenn.h"
#include "KESSPhotoAbsorbtion.h"
#include "KESSRelaxation.h"
#include "KField.h"
#include "KSIntScatteringBuilder.h"


using namespace Kassiopeia;
namespace katrin
{

class KSIntCalculatorKESSSet : public KSIntCalculatorSet
{
  public:
    KSIntCalculatorKESSSet();
    ~KSIntCalculatorKESSSet() override;

  public:
    void AddCalculator(KSIntCalculator* aCalculator) override;
    void ReleaseCalculators(KSIntScattering* aScattering) override;

  private:
    K_SET_GET(std::string, Name)
    K_SET_GET(bool, Elastic)
    K_SET_GET(std::string, Inelastic)
    K_SET_GET(bool, PhotoAbsorbtion)
    K_SET_GET(bool, AugerRelaxation)
    std::vector<KSIntCalculator*> fCalculators;
};

typedef KComplexElement<KSIntCalculatorKESSSet> KSIntCalculatorKESSSetBuilder;

template<> inline bool KSIntCalculatorKESSSetBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        aContainer->CopyTo(fObject, &KSIntCalculatorKESSSet::SetName);
        return true;
    }
    if (aContainer->GetName() == "elastic") {
        aContainer->CopyTo(fObject, &KSIntCalculatorKESSSet::SetElastic);
        return true;
    }
    if (aContainer->GetName() == "inelastic") {
        aContainer->CopyTo(fObject, &KSIntCalculatorKESSSet::SetInelastic);
        return true;
    }
    if (aContainer->GetName() == "photo_absorbtion") {
        aContainer->CopyTo(fObject, &KSIntCalculatorKESSSet::SetPhotoAbsorbtion);
        return true;
    }
    if (aContainer->GetName() == "auger_relaxation") {
        aContainer->CopyTo(fObject, &KSIntCalculatorKESSSet::SetAugerRelaxation);
        return true;
    }
    return false;
}

template<> inline bool KSIntCalculatorKESSSetBuilder::End()
{
    if (fObject->GetElastic() == true) {
        KSIntCalculator* aIntCalculator;

        aIntCalculator = new KESSElasticElsepa();
        aIntCalculator->SetName(fObject->GetName() + "_elastic");
        aIntCalculator->SetTag(fObject->GetName());
        fObject->AddCalculator(aIntCalculator);
    }

    if (fObject->GetInelastic() == "bethe_fano") {
        KSIntCalculator* aIntCalculator;

        aIntCalculator = new KESSInelasticBetheFano();
        if (fObject->GetPhotoAbsorbtion()) {
            auto* aPhotoabsorption = new KESSPhotoAbsorbtion();
            ((KESSInelasticBetheFano*) aIntCalculator)->SetIonisationCalculator(aPhotoabsorption);
        }
        if (fObject->GetAugerRelaxation()) {
            auto* aRelaxationCalculator = new KESSRelaxation();
            ((KESSInelasticBetheFano*) aIntCalculator)->SetRelaxationCalculator(aRelaxationCalculator);
        }

        aIntCalculator->SetName(fObject->GetName() + "_inelastic");
        aIntCalculator->SetTag(fObject->GetName());
        fObject->AddCalculator(aIntCalculator);
    }
    else if (fObject->GetInelastic() == "penn") {

        KSIntCalculator* aIntCalculator;

        aIntCalculator = new KESSInelasticPenn();
        if (fObject->GetPhotoAbsorbtion()) {
            auto* aPhotoabsorption = new KESSPhotoAbsorbtion();
            ((KESSInelasticPenn*) aIntCalculator)->SetIonisationCalculator(aPhotoabsorption);
        }
        if (fObject->GetAugerRelaxation()) {
            auto* aRelaxationCalculator = new KESSRelaxation();
            ((KESSInelasticPenn*) aIntCalculator)->SetRelaxationCalculator(aRelaxationCalculator);
        }

        aIntCalculator->SetName(fObject->GetName() + "_inelastic");
        aIntCalculator->SetTag(fObject->GetName());
        fObject->AddCalculator(aIntCalculator);
    }

    return true;
}
}  // namespace katrin

#endif  //Kassiopeia_KSIntCalculatorKESSBuilder_h_
