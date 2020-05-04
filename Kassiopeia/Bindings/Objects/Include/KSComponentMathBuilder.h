#ifndef Kassiopeia_KSComponentMathBuilder_h_
#define Kassiopeia_KSComponentMathBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentGroup.h"
#include "KSComponentMath.h"
#include "KSObjectsMessage.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

class KSComponentMathData
{
  public:
    std::string fName;
    std::string fGroupName;
    std::string fTerm;
    std::vector<std::string> fParents;
};

KSComponent* BuildOutputMath(std::vector<KSComponent*> aComponents, std::string aTerm)
{
    if (aComponents.at(0)->Is<unsigned short>() == true) {
        std::vector<unsigned short*> tComponents;
        tComponents.push_back(aComponents.at(0)->As<unsigned short>());
        for (size_t tIndex = 1; tIndex < aComponents.size(); tIndex++) {
            if (aComponents.at(tIndex)->Is<unsigned short>() == true) {
                tComponents.push_back(aComponents.at(tIndex)->As<unsigned short>());
            }
            else {
                objctmsg(eError) << "KSComponentMath does only support same types for all parents" << eom;
                return nullptr;
            }
        }
        return new KSComponentMath<unsigned short>(aComponents, tComponents, aTerm);
    }

    if (aComponents.at(0)->Is<short>() == true) {
        std::vector<short*> tComponents;
        tComponents.push_back(aComponents.at(0)->As<short>());
        for (size_t tIndex = 1; tIndex < aComponents.size(); tIndex++) {
            if (aComponents.at(tIndex)->Is<short>() == true) {
                tComponents.push_back(aComponents.at(tIndex)->As<short>());
            }
            else {
                objctmsg(eError) << "KSComponentMath does only support same types for all parents" << eom;
                return nullptr;
            }
        }
        return new KSComponentMath<short>(aComponents, tComponents, aTerm);
    }

    if (aComponents.at(0)->Is<unsigned int>() == true) {
        std::vector<unsigned int*> tComponents;
        tComponents.push_back(aComponents.at(0)->As<unsigned int>());
        for (size_t tIndex = 1; tIndex < aComponents.size(); tIndex++) {
            if (aComponents.at(tIndex)->Is<unsigned int>() == true) {
                tComponents.push_back(aComponents.at(tIndex)->As<unsigned int>());
            }
            else {
                objctmsg(eError) << "KSComponentMath does only support same types for all parents" << eom;
                return nullptr;
            }
        }
        return new KSComponentMath<unsigned int>(aComponents, tComponents, aTerm);
    }

    if (aComponents.at(0)->Is<int>() == true) {
        std::vector<int*> tComponents;
        tComponents.push_back(aComponents.at(0)->As<int>());
        for (size_t tIndex = 1; tIndex < aComponents.size(); tIndex++) {
            if (aComponents.at(tIndex)->Is<int>() == true) {
                tComponents.push_back(aComponents.at(tIndex)->As<int>());
            }
            else {
                objctmsg(eError) << "KSComponentMath does only support same types for all parents" << eom;
                return nullptr;
            }
        }
        return new KSComponentMath<int>(aComponents, tComponents, aTerm);
    }

    if (aComponents.at(0)->Is<unsigned long>() == true) {
        std::vector<unsigned long*> tComponents;
        tComponents.push_back(aComponents.at(0)->As<unsigned long>());
        for (size_t tIndex = 1; tIndex < aComponents.size(); tIndex++) {
            if (aComponents.at(tIndex)->Is<unsigned long>() == true) {
                tComponents.push_back(aComponents.at(tIndex)->As<unsigned long>());
            }
            else {
                objctmsg(eError) << "KSComponentMath does only support same types for all parents" << eom;
                return nullptr;
            }
        }
        return new KSComponentMath<unsigned long>(aComponents, tComponents, aTerm);
    }

    if (aComponents.at(0)->Is<long>() == true) {
        std::vector<long*> tComponents;
        tComponents.push_back(aComponents.at(0)->As<long>());
        for (size_t tIndex = 1; tIndex < aComponents.size(); tIndex++) {
            if (aComponents.at(tIndex)->Is<long>() == true) {
                tComponents.push_back(aComponents.at(tIndex)->As<long>());
            }
            else {
                objctmsg(eError) << "KSComponentMath does only support same types for all parents" << eom;
                return nullptr;
            }
        }
        return new KSComponentMath<long>(aComponents, tComponents, aTerm);
    }

    if (aComponents.at(0)->Is<float>() == true) {
        std::vector<float*> tComponents;
        tComponents.push_back(aComponents.at(0)->As<float>());
        for (size_t tIndex = 1; tIndex < aComponents.size(); tIndex++) {
            if (aComponents.at(tIndex)->Is<float>() == true) {
                tComponents.push_back(aComponents.at(tIndex)->As<float>());
            }
            else {
                objctmsg(eError) << "KSComponentMath does only support same types for all parents" << eom;
                return nullptr;
            }
        }
        return new KSComponentMath<float>(aComponents, tComponents, aTerm);
    }

    if (aComponents.at(0)->Is<double>() == true) {
        std::vector<double*> tComponents;
        tComponents.push_back(aComponents.at(0)->As<double>());
        for (size_t tIndex = 1; tIndex < aComponents.size(); tIndex++) {
            if (aComponents.at(tIndex)->Is<double>() == true) {
                tComponents.push_back(aComponents.at(tIndex)->As<double>());
            }
            else {
                objctmsg(eError) << "KSComponentMath does only support same types for all parents" << eom;
                return nullptr;
            }
        }
        return new KSComponentMath<double>(aComponents, tComponents, aTerm);
    }

    objctmsg(eError) << "KSComponentMathBuilder does only support int and double like types" << eom;
    return nullptr;
}

typedef KComplexElement<KSComponentMathData> KSComponentMathBuilder;

template<> inline bool KSComponentMathBuilder::Begin()
{
    fObject = new KSComponentMathData;
    return true;
}

template<> inline bool KSComponentMathBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string tName = aContainer->AsReference<std::string>();
        fObject->fName = tName;
        return true;
    }
    if (aContainer->GetName() == "group") {
        std::string tGroupName = aContainer->AsReference<std::string>();
        fObject->fGroupName = tGroupName;
        return true;
    }
    if (aContainer->GetName() == "term") {
        std::string tTerm = aContainer->AsReference<std::string>();
        fObject->fTerm = tTerm;
        return true;
    }
    if (aContainer->GetName() == "component") {
        objctmsg(eWarning)
            << "deprecated warning in KSComponentMathBuilder: Please use the attribute <parent> instead <component>"
            << eom;
        std::string tComponent = aContainer->AsReference<std::string>();
        fObject->fParents.push_back(tComponent);
        return true;
    }
    if (aContainer->GetName() == "parent") {
        std::string tComponent = aContainer->AsReference<std::string>();
        fObject->fParents.push_back(tComponent);
        return true;
    }
    return false;
}

template<> inline bool KSComponentMathBuilder::End()
{
    std::vector<KSComponent*> tParentComponents;
    if (!fObject->fGroupName.empty()) {
        auto* tComponentGroup = KToolbox::GetInstance().Get<KSComponentGroup>(fObject->fGroupName);
        for (size_t tNameIndex = 0; tNameIndex < fObject->fParents.size(); tNameIndex++) {
            KSComponent* tOneComponent = nullptr;
            for (unsigned int tGroupIndex = 0; tGroupIndex < tComponentGroup->ComponentCount(); tGroupIndex++) {
                KSComponent* tGroupComponent = tComponentGroup->ComponentAt(tGroupIndex);
                if (tGroupComponent->GetName() == fObject->fParents.at(tNameIndex)) {
                    tOneComponent = tGroupComponent;
                    break;
                }
            }
            if (tOneComponent == nullptr) {
                objctmsg(eError) << "KSComponentMathBuilder can not find component < "
                                 << fObject->fParents.at(tNameIndex) << " > in group < " << fObject->fGroupName << " >"
                                 << eom;
            }
            tParentComponents.push_back(tOneComponent);
        }
    }
    else {
        for (size_t tIndex = 0; tIndex < fObject->fParents.size(); tIndex++) {
            auto* tOneComponent = KToolbox::GetInstance().Get<KSComponent>(fObject->fParents.at(tIndex));
            tParentComponents.push_back(tOneComponent);
        }
    }
    KSComponent* tComponent = BuildOutputMath(tParentComponents, fObject->fTerm);
    tComponent->SetName(fObject->fName);
    delete fObject;
    Set(tComponent);
    return true;
}

}  // namespace katrin

#endif
