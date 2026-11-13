#ifndef Kassiopeia_KSCommandMemberBuilder_h_
#define Kassiopeia_KSCommandMemberBuilder_h_

#include "KComplexElement.hh"
#include "KSCommandMember.h"
#include "KSObjectsMessage.h"
#include "KToolbox.h"

using namespace Kassiopeia;
namespace katrin
{

class KSCommandMemberData
{
  public:
    std::string fName;
    std::string fFieldName;
    std::string fParentName;
    std::string fChildName;
};

//! command object encapsulating a request to combine or change simulation objects.
//! given an instance of a kassiopeia object, a string can be given to it to request a command.
//! if a matching association between the string and a member function of the parent's class is found, a command is created which binds the parent, its member, and a given child object together.
//! typically commands come in add/remove or set/clear pairs which compose objects together as in the example, or get/set pairs by which an object's configuration may be altered.
//! @example <ks_command_member name="command_remove_cyclotron" parent="trajectory_exact" field="remove_control" child="control_cyclotron"/>
typedef KComplexElement<KSCommandMemberData> KSCommandMemberBuilder;

template<> inline bool KSCommandMemberBuilder::Begin()
{
    fObject = new KSCommandMemberData;
    return true;
}

template<> inline bool KSCommandMemberBuilder::AddAttribute(KContainer* aContainer)
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
    if (aContainer->GetName() == "child") {
        std::string tChild = aContainer->AsString();
        fObject->fChildName = tChild;
        return true;
    }
    if (aContainer->GetName() == "field") {
        std::string tField = aContainer->AsString();
        fObject->fFieldName = tField;
        return true;
    }
    return false;
}

template<> inline bool KSCommandMemberBuilder::End()
{
    objctmsg_debug("Command: " << fObject->fParentName << " . " << fObject->fChildName << " : " << fObject->fFieldName
                               << eom);
    auto* tParent = KToolbox::GetInstance().Get<KSComponent>(fObject->fParentName);
    if (tParent == nullptr) {
        objctmsg(eError) << "command member <" << fObject->fName << "> could not find parent <" << fObject->fParentName
                         << ">" << eom;
        return false;
    }

    auto* tChild = KToolbox::GetInstance().Get<KSComponent>(fObject->fChildName);
    if (tChild == nullptr) {
        tChild = KToolbox::GetInstance().Get<KSComponent>(fObject->fChildName + "_");  // try alias
        if (tChild == nullptr) {
            objctmsg(eError) << "command member <" << fObject->fName << "> could not find child <"
                             << fObject->fChildName << ">" << eom;
            return false;
        }
    }

    KSCommand* tCommand = tParent->Command(fObject->fFieldName, tChild);
    if (tCommand == nullptr) {
        objctmsg(eError) << "command member <" << fObject->fName << "> could not activate field <"
                         << fObject->fFieldName << "> in parent <" << fObject->fParentName << ">" << eom;
        return false;
    }

    if (!fObject->fName.empty()) {
        tCommand->SetName(fObject->fName);
    }
    delete fObject;
    Set(tCommand);
    return true;
}

}  // namespace katrin

#endif
