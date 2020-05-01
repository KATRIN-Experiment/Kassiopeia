#ifndef Kassiopeia_KSComponentDeltaBuilder_h_
#define Kassiopeia_KSComponentDeltaBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentDelta.h"
#include "KSComponentGroup.h"
#include "KSObjectsMessage.h"
#include "KToolbox.h"
#include "KTwoVector.hh"
using KGeoBag::KTwoVector;

#include "KThreeVector.hh"
using KGeoBag::KThreeVector;

#include "KTwoMatrix.hh"
using KGeoBag::KTwoMatrix;

#include "KThreeMatrix.hh"
using KGeoBag::KThreeMatrix;

using namespace Kassiopeia;
namespace katrin
{

class KSComponentDeltaData
{
  public:
    std::string fName;
    std::string fGroupName;
    std::string fParentName;
};

KSComponent* BuildOutputDelta(KSComponent* aComponent)
{
    if (aComponent->Is<bool>() == true) {
        return new KSComponentDelta<bool>(aComponent, aComponent->As<bool>());
    }

    if (aComponent->Is<unsigned char>() == true) {
        return new KSComponentDelta<unsigned char>(aComponent, aComponent->As<unsigned char>());
    }

    if (aComponent->Is<char>() == true) {
        return new KSComponentDelta<char>(aComponent, aComponent->As<char>());
    }

    if (aComponent->Is<unsigned short>() == true) {
        return new KSComponentDelta<unsigned short>(aComponent, aComponent->As<unsigned short>());
    }

    if (aComponent->Is<short>() == true) {
        return new KSComponentDelta<short>(aComponent, aComponent->As<short>());
    }

    if (aComponent->Is<unsigned int>() == true) {
        return new KSComponentDelta<unsigned int>(aComponent, aComponent->As<unsigned int>());
    }

    if (aComponent->Is<int>() == true) {
        return new KSComponentDelta<int>(aComponent, aComponent->As<int>());
    }

    if (aComponent->Is<unsigned long>() == true) {
        return new KSComponentDelta<unsigned long>(aComponent, aComponent->As<unsigned long>());
    }

    if (aComponent->Is<long>() == true) {
        return new KSComponentDelta<long>(aComponent, aComponent->As<long>());
    }

    if (aComponent->Is<float>() == true) {
        return new KSComponentDelta<float>(aComponent, aComponent->As<float>());
    }

    if (aComponent->Is<double>() == true) {
        return new KSComponentDelta<double>(aComponent, aComponent->As<double>());
    }

    if (aComponent->Is<KTwoVector>() == true) {
        return new KSComponentDelta<KTwoVector>(aComponent, aComponent->As<KTwoVector>());
    }

    if (aComponent->Is<KThreeVector>() == true) {
        return new KSComponentDelta<KThreeVector>(aComponent, aComponent->As<KThreeVector>());
    }

    if (aComponent->Is<KTwoMatrix>() == true) {
        return new KSComponentDelta<KTwoMatrix>(aComponent, aComponent->As<KTwoMatrix>());
    }

    if (aComponent->Is<KThreeMatrix>() == true) {
        return new KSComponentDelta<KThreeMatrix>(aComponent, aComponent->As<KThreeMatrix>());
    }

    return nullptr;
}

typedef KComplexElement<KSComponentDeltaData> KSComponentDeltaBuilder;

template<> inline bool KSComponentDeltaBuilder::Begin()
{
    fObject = new KSComponentDeltaData;
    return true;
}

template<> inline bool KSComponentDeltaBuilder::AddAttribute(KContainer* aContainer)
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
    if (aContainer->GetName() == "component") {
        objctmsg(eWarning)
            << "deprecated warning in KSComponentDeltaBuilder: Please use the attribute <parent> instead <component>"
            << eom;
        std::string tParentName = aContainer->AsReference<std::string>();
        fObject->fParentName = tParentName;
        return true;
    }
    if (aContainer->GetName() == "parent") {
        std::string tParentName = aContainer->AsReference<std::string>();
        fObject->fParentName = tParentName;
        return true;
    }
    return false;
}

template<> inline bool KSComponentDeltaBuilder::End()
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
            objctmsg(eError) << "component delta builder could not find component <" << fObject->fParentName
                             << "> in group <" << fObject->fGroupName << ">" << eom;
            return false;
        }
    }
    else {
        tParentComponent = KToolbox::GetInstance().Get<KSComponent>(fObject->fParentName);
    }
    KSComponent* tComponent = BuildOutputDelta(tParentComponent);
    tComponent->SetName(fObject->fName);
    delete fObject;
    Set(tComponent);
    return true;
}

}  // namespace katrin

#endif
