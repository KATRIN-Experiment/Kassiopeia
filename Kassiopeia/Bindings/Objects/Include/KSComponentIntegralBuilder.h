#ifndef Kassiopeia_KSComponentIntegralBuilder_h_
#define Kassiopeia_KSComponentIntegralBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentGroup.h"
#include "KSComponentIntegral.h"
#include "KSObjectsMessage.h"
#include "KToolbox.h"
#include "KTwoVector.hh"
#include "KThreeVector.hh"
#include "KTwoMatrix.hh"
#include "KThreeMatrix.hh"

using namespace Kassiopeia;
namespace katrin
{

class KSComponentIntegralData
{
  public:
    std::string fName;
    std::string fGroupName;
    std::string fParentName;
};

inline KSComponent* BuildOutputIntegral(KSComponent* aComponent)
{
    if (aComponent->Is<bool>() == true) {
        return new KSComponentIntegral<bool>(aComponent, aComponent->As<bool>());
    }

    if (aComponent->Is<unsigned char>() == true) {
        return new KSComponentIntegral<unsigned char>(aComponent, aComponent->As<unsigned char>());
    }

    if (aComponent->Is<char>() == true) {
        return new KSComponentIntegral<char>(aComponent, aComponent->As<char>());
    }

    if (aComponent->Is<unsigned short>() == true) {
        return new KSComponentIntegral<unsigned short>(aComponent, aComponent->As<unsigned short>());
    }

    if (aComponent->Is<short>() == true) {
        return new KSComponentIntegral<short>(aComponent, aComponent->As<short>());
    }

    if (aComponent->Is<unsigned int>() == true) {
        return new KSComponentIntegral<unsigned int>(aComponent, aComponent->As<unsigned int>());
    }

    if (aComponent->Is<int>() == true) {
        return new KSComponentIntegral<int>(aComponent, aComponent->As<int>());
    }

    if (aComponent->Is<unsigned long>() == true) {
        return new KSComponentIntegral<unsigned long>(aComponent, aComponent->As<unsigned long>());
    }

    if (aComponent->Is<long>() == true) {
        return new KSComponentIntegral<long>(aComponent, aComponent->As<long>());
    }

    if (aComponent->Is<float>() == true) {
        return new KSComponentIntegral<float>(aComponent, aComponent->As<float>());
    }

    if (aComponent->Is<double>() == true) {
        return new KSComponentIntegral<double>(aComponent, aComponent->As<double>());
    }

    if (aComponent->Is<KTwoVector>() == true) {
        return new KSComponentIntegral<KTwoVector>(aComponent, aComponent->As<KTwoVector>());
    }

    if (aComponent->Is<KThreeVector>() == true) {
        return new KSComponentIntegral<KThreeVector>(aComponent, aComponent->As<KThreeVector>());
    }

    if (aComponent->Is<KTwoMatrix>() == true) {
        return new KSComponentIntegral<KTwoMatrix>(aComponent, aComponent->As<KTwoMatrix>());
    }

    if (aComponent->Is<KThreeMatrix>() == true) {
        return new KSComponentIntegral<KThreeMatrix>(aComponent, aComponent->As<KThreeMatrix>());
    }

    return nullptr;
}

typedef KComplexElement<KSComponentIntegralData> KSComponentIntegralBuilder;

template<> inline bool KSComponentIntegralBuilder::Begin()
{
    fObject = new KSComponentIntegralData;
    return true;
}

template<> inline bool KSComponentIntegralBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string tName = aContainer->AsString();
        fObject->fName = tName;
        return true;
    }
    if (aContainer->GetName() == "group") {
        std::string tGroupName = aContainer->AsString();
        fObject->fGroupName = tGroupName;
        return true;
    }
    if (aContainer->GetName() == "component") {
        objctmsg(eWarning)
            << "deprecated warning in KSComponentIntegralBuilder: Please use the attribute <parent> instead <component>"
            << eom;
        std::string tParentName = aContainer->AsString();
        fObject->fParentName = tParentName;
        return true;
    }
    if (aContainer->GetName() == "parent") {
        std::string tParentName = aContainer->AsString();
        fObject->fParentName = tParentName;
        return true;
    }
    return false;
}

template<> inline bool KSComponentIntegralBuilder::End()
{
    KSComponent* tParentComponent = nullptr;
    if (fObject->fGroupName.empty() == false) {
        auto* tComponentGroup = KToolbox::GetInstance().Get<KSComponentGroup>(fObject->fGroupName);
        for (unsigned int tIndex = 0; tIndex < tComponentGroup->ComponentCount(); tIndex++) {
            KSComponent* tGroupComponent = tComponentGroup->ComponentAt(tIndex);
            if (tGroupComponent->GetName() == fObject->fParentName) {
                tParentComponent = tGroupComponent;
                break;
            }
        }
        if (tParentComponent == nullptr) {
            objctmsg(eError) << "component integral builder could not find component <" << fObject->fParentName
                             << "> in group <" << fObject->fGroupName << ">" << eom;
            return false;
        }
    }
    else {
        tParentComponent = KToolbox::GetInstance().Get<KSComponent>(fObject->fParentName);
    }
    KSComponent* tComponent = BuildOutputIntegral(tParentComponent);
    tComponent->SetName(fObject->fName);
    delete fObject;
    Set(tComponent);
    return true;
}

}  // namespace katrin

#endif
