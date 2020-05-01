#ifndef Kassiopeia_KSComponentTemplate_h_
#define Kassiopeia_KSComponentTemplate_h_

#include "KSCommandMember.h"
#include "KSComponent.h"
#include "KSComponentMember.h"

namespace Kassiopeia
{

template<class XThisType, class XParentOne = void, class XParentTwo = void, class XParentThree = void>
class KSComponentTemplate;

//******************
//1-parent component
//******************

template<class XThisType, class XFirstParentType>
class KSComponentTemplate<XThisType, XFirstParentType, void, void> : virtual public KSComponent, public XFirstParentType
{
  public:
    KSComponentTemplate()
    {
        Set(static_cast<XThisType*>(this));
    }
    ~KSComponentTemplate() override {}

    //***********
    //KSComponent
    //***********

  public:
    KSComponent* Component(const std::string& aField) override
    {
        objctmsg_debug("component <" << this->GetName() << "> building output named <" << aField << ">" << eom)
            KSComponent* tComponent = KSDictionary<XThisType>::GetComponent(this, aField);
        if (tComponent == nullptr) {
            return XFirstParentType::Component(aField);
        }
        else {
            fChildComponents.push_back(tComponent);
            return tComponent;
        }
    }
    KSCommand* Command(const std::string& aField, KSComponent* aChild) override
    {
        if (aChild == nullptr) {
            objctmsg(eError) << "component <" << this->GetName() << "> could not build command named <" << aField
                             << "> (invalid child component)" << eom;
            return nullptr;
        }

        objctmsg_debug("component <" << this->GetName() << "> building command named <" << aField << ">" << eom)
            KSCommand* tCommand = KSDictionary<XThisType>::GetCommand(this, aChild, aField);
        if (tCommand == nullptr) {
            return XFirstParentType::Command(aField, aChild);
        }
        else {
            return tCommand;
        }
    }
};

//******************
//0-parent component
//******************

template<class XThisType> class KSComponentTemplate<XThisType, void, void, void> : virtual public KSComponent
{
  public:
    KSComponentTemplate() : KSComponent()
    {
        Set(static_cast<XThisType*>(this));
    }
    ~KSComponentTemplate() override {}

    //***********
    //KSComponent
    //***********

  public:
    KSComponent* Component(const std::string& aLabel) override
    {
        objctmsg_debug("component <" << this->GetName() << "> building component named <" << aLabel << ">" << eom)
            KSComponent* tComponent = KSDictionary<XThisType>::GetComponent(this, aLabel);
        if (tComponent == nullptr) {
            objctmsg(eError) << "component <" << this->GetName() << "> has no component named <" << aLabel << ">"
                             << eom;
            return nullptr;
        }
        else {
            fChildComponents.push_back(tComponent);
            return tComponent;
        }
    }
    KSCommand* Command(const std::string& aField, KSComponent* aChild) override
    {
        objctmsg_debug("component <" << this->GetName() << "> building command named <" << aField << ">" << eom)
            KSCommand* tCommand = KSDictionary<XThisType>::GetCommand(this, aChild, aField);
        if (tCommand == nullptr) {
            objctmsg(eError) << "component <" << this->GetName() << "> has no command named <" << aField << ">" << eom;
            return nullptr;
        }
        else {
            return tCommand;
        }
    }
};

}  // namespace Kassiopeia

#endif
