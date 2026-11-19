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

//! component object encapsulating the extraction of a component from within a composite.
//! given an instance of a kassiopeia object, a string can be given to it to request a component.
//! if a matching association between the string and a member function of the parent's class is found, a component is created which binds the result of calling the member function of the parent into a kassiopeia object.
//! typically components are built from getter functions, and may be chained together by using another component as parent, as in the example.
//! @example
//! <output name="step_final_particle" parent="step" field="final_particle"/>
//! <output name="step_final_particle_position" parent="step_final_particle" field="position"/>
//! <output name="step_final_particle_position_z" parent="step_final_particle_position" field="z"/>
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
