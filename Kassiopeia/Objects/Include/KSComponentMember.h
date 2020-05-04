#ifndef Kassiopeia_KSComponentMember_h_
#define Kassiopeia_KSComponentMember_h_

#include "KSDictionary.h"

#include <typeinfo>

namespace Kassiopeia
{

template<class XMemberType> class KSComponentMember;

template<class XMemberType> class KSComponentMemberFactory;

//**********************
//const-reference getter
//**********************

template<class XParentType, class XValueType>
class KSComponentMember<const XValueType& (XParentType::*) (void) const> : public KSComponent
{
  public:
    KSComponentMember(KSComponent* aParentComponent, XParentType* aParentPointer,
                      const XValueType& (XParentType::*aMember)() const) :
        KSComponent(),
        fTarget(aParentPointer),
        fMember(aMember),
        fValue()
    {
        Set(&fValue);
        this->SetParent(aParentComponent);
    }
    KSComponentMember(const KSComponentMember<const XValueType& (XParentType::*) (void) const>& aCopy) :
        KSComponent(aCopy),
        fTarget(aCopy.fTarget),
        fMember(aCopy.fMember),
        fValue(aCopy.fValue)
    {
        Set(&fValue);
        this->SetParent(aCopy.fParentComponent);
    }
    ~KSComponentMember() override {}

    //***********
    //KSComponent
    //***********

  public:
    KSComponent* Clone() const override
    {
        return new KSComponentMember<const XValueType& (XParentType::*) (void) const>(*this);
    }
    KSComponent* Component(const std::string& aField) override
    {
        objctmsg_debug("const-reference component member <" << this->GetName() << "> building component named <"
                                                            << aField << ">" << eom) KSComponent* tComponent =
            KSDictionary<XValueType>::GetComponent(this, aField);
        if (tComponent == nullptr) {
            objctmsg(eError) << "const-reference component member <" << this->GetName() << "> has no output named <"
                             << aField << ">" << eom;
        }
        else {
            fChildComponents.push_back(tComponent);
        }
        return tComponent;
    }
    KSCommand* Command(const std::string& /*aField*/, KSComponent* /*aChild*/) override
    {
        return nullptr;
    }

  public:
    void PushUpdateComponent() override
    {
        objctmsg_debug("const-reference component member <" << this->GetName() << "> pushing update" << eom);
        fValue = (fTarget->*fMember)();
        return;
    }
    void PullUpdateComponent() override
    {
        objctmsg_debug("const-reference component member <" << this->GetName() << "> pulling update" << eom);
        fValue = (fTarget->*fMember)();
        return;
    }

  private:
    XParentType* fTarget;
    const XValueType& (XParentType::*fMember)() const;
    XValueType fValue;
};

template<class XParentType, class XValueType>
class KSComponentMemberFactory<const XValueType& (XParentType::*) () const> : public KSComponentFactory
{
  public:
    KSComponentMemberFactory(const XValueType& (XParentType::*aMember)() const) : fMember(aMember) {}
    ~KSComponentMemberFactory() override {}

  public:
    KSComponent* CreateComponent(KSComponent* aParent) const override
    {
        auto* tParent = aParent->As<XParentType>();
        if (tParent == nullptr) {
            objctmsg_debug("  component parent <" << aParent->GetName() << "> could not be cast to type <"
                                                  << typeid(XParentType).name() << ">" << eom);
            return nullptr;
        }

        objctmsg_debug("  component built" << eom);
        return new KSComponentMember<const XValueType& (XParentType::*) () const>(aParent, tParent, fMember);
    }

  private:
    const XValueType& (XParentType::*fMember)() const;
};

//*****************
//copy-value getter
//*****************

template<class XParentType, class XValueType>
class KSComponentMember<XValueType (XParentType::*)(void) const> : public KSComponent
{
  public:
    KSComponentMember(KSComponent* aParentComponent, XParentType* aParentPointer,
                      XValueType (XParentType::*aMember)() const) :
        KSComponent(),
        fTarget(aParentPointer),
        fMember(aMember),
        fValue()
    {
        Set(&fValue);
        fParentComponent = aParentComponent;
    }
    KSComponentMember(const KSComponentMember<XValueType (XParentType::*)(void) const>& aCopy) :
        KSComponent(aCopy),
        fTarget(aCopy.fTarget),
        fMember(aCopy.fMember),
        fValue(aCopy.fValue)
    {
        Set(&fValue);
        fParentComponent = aCopy.fParentComponent;
    }
    ~KSComponentMember() override {}

    //***********
    //KSComponent
    //***********

  public:
    KSComponent* Clone() const override
    {
        return new KSComponentMember<XValueType (XParentType::*)(void) const>(*this);
    }
    KSComponent* Component(const std::string& aField) override
    {
        objctmsg_debug("copy-value component member <" << this->GetName() << "> building component named <" << aField
                                                       << ">" << eom) KSComponent* tComponent =
            KSDictionary<XValueType>::GetComponent(this, aField);
        if (tComponent == nullptr) {
            objctmsg(eError) << "const-reference component member <" << this->GetName() << "> has no output named <"
                             << aField << ">" << eom;
        }
        else {
            fChildComponents.push_back(tComponent);
        }
        return tComponent;
    }
    KSCommand* Command(const std::string& /*aField*/, KSComponent* /*aChild*/) override
    {
        return nullptr;
    }

  protected:
    void PushUpdateComponent() override
    {
        objctmsg_debug("copy-value component member <" << this->GetName() << "> pushing update" << eom);
        fValue = (fTarget->*fMember)();
        return;
    }

  private:
    XParentType* fTarget;
    XValueType (XParentType::*fMember)() const;
    XValueType fValue;
};

template<class XParentType, class XValueType>
class KSComponentMemberFactory<XValueType (XParentType::*)() const> : public KSComponentFactory
{
  public:
    KSComponentMemberFactory(XValueType (XParentType::*aMember)() const) : fMember(aMember) {}
    ~KSComponentMemberFactory() override {}

  public:
    KSComponent* CreateComponent(KSComponent* aParent) const override
    {
        auto* tParent = aParent->As<XParentType>();
        if (tParent == nullptr) {
            objctmsg_debug("  component parent <" << aParent->GetName() << "> could not be cast to type <"
                                                  << typeid(XParentType).name() << ">" << eom);
            return nullptr;
        }

        objctmsg_debug("  component built" << eom);
        return new KSComponentMember<XValueType (XParentType::*)() const>(aParent, tParent, fMember);
    }

  private:
    XValueType (XParentType::*fMember)() const;
};

//**********
//dictionary
//**********

template<class XType>
template<class XMemberType>
int KSDictionary<XType>::AddComponent(XMemberType aMember, const std::string& aLabel)
{
    if (fComponentFactories == nullptr) {
        fComponentFactories = new ComponentFactoryMap();
    }

    auto* tComponentMemberFactory = new KSComponentMemberFactory<XMemberType>(aMember);
    fComponentFactories->insert(ComponentFactoryEntry(aLabel, tComponentMemberFactory));
    return 0;
}

}  // namespace Kassiopeia

#endif
