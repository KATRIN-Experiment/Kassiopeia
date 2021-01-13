#ifndef Kassiopeia_KSComponentMemberBuilder_h_
#define Kassiopeia_KSComponentMemberBuilder_h_

#include "KComplexElement.hh"
#include "KSComponentMember.h"
#include "KSObjectsMessage.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

class KSComponentMemberData
{
  public:
    std::string fName;
    std::string fFieldName;
    std::string fParentName;
};

typedef KComplexElement<KSComponentMemberData> KSComponentBuilder;

template<> inline bool KSComponentBuilder::Begin()
{
    fObject = new KSComponentMemberData;
    return true;
}

template<> inline bool KSComponentBuilder::AddAttribute(KContainer* aContainer)
{
    if (aContainer->GetName() == "name") {
        std::string tName = aContainer->AsString();
        fObject->fName = tName;
        return true;
    }
    if (aContainer->GetName() == "parent") {
        std::string tParent = aContainer->AsString();
        fObject->fParentName = tParent;
        return true;
    }
    if (aContainer->GetName() == "field") {
        std::string tField = aContainer->AsString();
        fObject->fFieldName = tField;
        return true;
    }
    return false;
}

template<> inline bool KSComponentBuilder::End()
{
    auto* tParent = KToolbox::GetInstance().Get<KSComponent>(fObject->fParentName);
    if (tParent == nullptr) {
        objctmsg(eError) << "component member <" << fObject->fName << "> could not find parent <"
                         << fObject->fParentName << ">" << eom;
    }
    KSComponent* tComponent = tParent->Component(fObject->fFieldName);
    if (tComponent == nullptr) {
        objctmsg(eError) << "component member <" << fObject->fName << "> could not find field <" << fObject->fFieldName
                         << "> with parent <" << fObject->fParentName << ">" << eom;
    }
    if (!fObject->fName.empty()) {
        tComponent->SetName(fObject->fName);
    }
    delete fObject;
    Set(tComponent);
    return true;
}

}  // namespace katrin

#endif
