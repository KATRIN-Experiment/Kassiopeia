#ifndef Kassiopeia_KSTermOutputBuilder_h_
#define Kassiopeia_KSTermOutputBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentGroup.h"
#include "KSTermOutput.h"
#include "KSTerminatorsMessage.h"
#include "KToolbox.h"

#include <limits>

using namespace Kassiopeia;
namespace katrin
{

class KSTermOutputData
{
  public:
    std::string fName;
    std::string fGroupName;
    std::string fComponentName;
    double fMinValue;
    double fMaxValue;
};

typedef KComplexElement<KSTermOutputData> KSTermOutputBuilder;

template<> inline bool KSTermOutputBuilder::Begin()
{
    fObject = new KSTermOutputData;
    fObject->fMinValue = -1.0 * std::numeric_limits<double>::max();
    fObject->fMaxValue = std::numeric_limits<double>::max();
    return true;
}

template<> inline bool KSTermOutputBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string tName = aContainer->AsReference<std::string>();
        fObject->fName = tName;
        return true;
    }
    if (aContainer->GetName() == "min_value") {
        double tValue = aContainer->AsReference<double>();
        fObject->fMinValue = tValue;
        return true;
    }
    if (aContainer->GetName() == "max_value") {
        double tValue = aContainer->AsReference<double>();
        fObject->fMaxValue = tValue;
        return true;
    }
    if (aContainer->GetName() == "group") {
        std::string tName = aContainer->AsReference<std::string>();
        fObject->fGroupName = tName;
        return true;
    }
    if (aContainer->GetName() == "component") {
        termmsg(eWarning)
            << "deprecated warning in KSTermOutputBuilder: Please use the attribute <output> instead <component>"
            << eom;
        std::string tName = aContainer->AsReference<std::string>();
        fObject->fComponentName = tName;
        return true;
    }
    if (aContainer->GetName() == "output") {
        std::string tName = aContainer->AsReference<std::string>();
        fObject->fComponentName = tName;
        return true;
    }
    return false;
}

template<> inline bool KSTermOutputBuilder::End()
{
    KSComponent* tComponent = nullptr;
    if (fObject->fGroupName.empty() == false) {
        auto* tComponentGroup = KToolbox::GetInstance().Get<KSComponentGroup>(fObject->fGroupName);
        for (unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++) {
            KSComponent* tGroupComponent = tComponentGroup->ComponentAt(tIndex);
            if (tGroupComponent->GetName() == fObject->fComponentName) {
                tComponent = tGroupComponent;
                break;
            }
        }
        if (tComponent == nullptr) {
            objctmsg(eError) << "term output builder could not find component <" << fObject->fComponentName
                             << "> in group <" << fObject->fGroupName << ">" << eom;
            return false;
        }
    }
    else {
        tComponent = KToolbox::GetInstance().Get<KSComponent>(fObject->fComponentName);
    }


    KSTerminator* tTerm = nullptr;

    if (tComponent->Is<unsigned short>() == true) {
        auto* tTermOutput = new KSTermOutput<unsigned short>();
        tTermOutput->SetName(fObject->fName);
        tTermOutput->SetMinValue(fObject->fMinValue);
        tTermOutput->SetMaxValue(fObject->fMaxValue);
        tTermOutput->SetValue(tComponent->As<unsigned short>());
        tTerm = tTermOutput;
        delete fObject;
        Set(tTerm);
        return true;
    }

    if (tComponent->Is<short>() == true) {
        auto* tTermOutput = new KSTermOutput<short>();
        tTermOutput->SetName(fObject->fName);
        tTermOutput->SetMinValue(fObject->fMinValue);
        tTermOutput->SetMaxValue(fObject->fMaxValue);
        tTermOutput->SetValue(tComponent->As<short>());
        tTerm = tTermOutput;
        delete fObject;
        Set(tTerm);
        return true;
    }

    if (tComponent->Is<unsigned int>() == true) {
        auto* tTermOutput = new KSTermOutput<unsigned int>();
        tTermOutput->SetName(fObject->fName);
        tTermOutput->SetMinValue(fObject->fMinValue);
        tTermOutput->SetMaxValue(fObject->fMaxValue);
        tTermOutput->SetValue(tComponent->As<unsigned int>());
        tTerm = tTermOutput;
        delete fObject;
        Set(tTerm);
        return true;
    }

    if (tComponent->Is<int>() == true) {
        auto* tTermOutput = new KSTermOutput<int>();
        tTermOutput->SetName(fObject->fName);
        tTermOutput->SetMinValue(fObject->fMinValue);
        tTermOutput->SetMaxValue(fObject->fMaxValue);
        tTermOutput->SetValue(tComponent->As<int>());
        tTerm = tTermOutput;
        delete fObject;
        Set(tTerm);
        return true;
    }

    if (tComponent->Is<unsigned long>() == true) {
        auto* tTermOutput = new KSTermOutput<unsigned long>();
        tTermOutput->SetName(fObject->fName);
        tTermOutput->SetMinValue(fObject->fMinValue);
        tTermOutput->SetMaxValue(fObject->fMaxValue);
        tTermOutput->SetValue(tComponent->As<unsigned long>());
        tTerm = tTermOutput;
        delete fObject;
        Set(tTerm);
        return true;
    }

    if (tComponent->Is<long>() == true) {
        auto* tTermOutput = new KSTermOutput<long>();
        tTermOutput->SetName(fObject->fName);
        tTermOutput->SetMinValue(fObject->fMinValue);
        tTermOutput->SetMaxValue(fObject->fMaxValue);
        tTermOutput->SetValue(tComponent->As<long>());
        tTerm = tTermOutput;
        delete fObject;
        Set(tTerm);
        return true;
    }

    if (tComponent->Is<float>() == true) {
        auto* tTermOutput = new KSTermOutput<float>();
        tTermOutput->SetName(fObject->fName);
        tTermOutput->SetMinValue(fObject->fMinValue);
        tTermOutput->SetMaxValue(fObject->fMaxValue);
        tTermOutput->SetValue(tComponent->As<float>());
        tTerm = tTermOutput;
        delete fObject;
        Set(tTerm);
        return true;
    }

    if (tComponent->Is<double>() == true) {
        auto* tTermOutput = new KSTermOutput<double>();
        tTermOutput->SetName(fObject->fName);
        tTermOutput->SetMinValue(fObject->fMinValue);
        tTermOutput->SetMaxValue(fObject->fMaxValue);
        tTermOutput->SetValue(tComponent->As<double>());
        tTerm = tTermOutput;
        delete fObject;
        Set(tTerm);
        return true;
    }

    objctmsg(eError) << "component in term output builder is of non supported type " << eom;
    return false;
}

}  // namespace katrin
#endif
