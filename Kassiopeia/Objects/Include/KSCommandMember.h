#ifndef Kassiopeia_KSCommandTemplate_h_
#define Kassiopeia_KSCommandTemplate_h_

#include "KSDictionary.h"

#include <typeinfo>

namespace Kassiopeia
{

template<class XParentType, class XChildType> class KSCommandMemberAdd : public KSCommand
{
  public:
    KSCommandMemberAdd(KSComponent* aParentComponent, XParentType* aParent, KSComponent* aChildComponent,
                       XChildType* aChild, void (XParentType::*anAddMember)(XChildType*),
                       void (XParentType::*aRemoveMember)(XChildType*)) :
        KSCommand(),
        fParentPointer(aParent),
        fChildPointer(aChild),
        fAddMember(anAddMember),
        fRemoveMember(aRemoveMember)
    {
        Set(this);
        fParentComponent = aParentComponent;
        fChildComponent = aChildComponent;
    }
    KSCommandMemberAdd(const KSCommandMemberAdd<XParentType, XChildType>& aCopy) :
        KSCommand(aCopy),
        fParentPointer(aCopy.fParentPointer),
        fChildPointer(aCopy.fChildPointer),
        fAddMember(aCopy.fAddMember),
        fRemoveMember(aCopy.fRemoveMember)

    {
        Set(this);
        fParentComponent = aCopy.fParentComponent;
        fChildComponent = aCopy.fChildComponent;
    }
    ~KSCommandMemberAdd() override {}

  public:
    KSCommandMemberAdd* Clone() const override
    {
        return new KSCommandMemberAdd<XParentType, XChildType>(*this);
    }

  protected:
    void ActivateCommand() override
    {
        (fParentPointer->*fAddMember)(fChildPointer);
        fChildComponent->Activate();
        return;
    }
    void DeactivateCommand() override
    {
        fChildComponent->Deactivate();
        (fParentPointer->*fRemoveMember)(fChildPointer);
        return;
    }

  private:
    XParentType* fParentPointer;
    XChildType* fChildPointer;
    void (XParentType::*fAddMember)(XChildType*);
    void (XParentType::*fRemoveMember)(XChildType*);
};

template<class XParentType, class XChildType> class KSCommandMemberAddFactory : public KSCommandFactory
{
  public:
    KSCommandMemberAddFactory(void (XParentType::*anAddMember)(XChildType*),
                              void (XParentType::*aRemoveMember)(XChildType*)) :
        fAddMember(anAddMember),
        fRemoveMember(aRemoveMember)
    {}
    ~KSCommandMemberAddFactory() override {}

  public:
    KSCommand* CreateCommand(KSComponent* aParent, KSComponent* aChild) const override
    {
        auto* tParent = aParent->As<XParentType>();
        if (tParent == nullptr) {
            objctmsg_debug("  command parent <" << aParent->GetName() << "> could not be cast to type <"
                                                << typeid(XParentType).name() << ">" << eom);
            return nullptr;
        }

        auto* tChild = aChild->As<XChildType>();
        if (tChild == nullptr) {
            objctmsg_debug("  command child <" << aChild->GetName() << "> could not be cast to type <"
                                               << typeid(XChildType).name() << ">" << eom);
            return nullptr;
        }

        objctmsg_debug("  command built" << eom);
        return new KSCommandMemberAdd<XParentType, XChildType>(aParent,
                                                               tParent,
                                                               aChild,
                                                               tChild,
                                                               fAddMember,
                                                               fRemoveMember);
    }

  private:
    void (XParentType::*fAddMember)(XChildType*);
    void (XParentType::*fRemoveMember)(XChildType*);
};


template<class XParentType, class XChildType> class KSCommandMemberRemove : public KSCommand
{
  public:
    KSCommandMemberRemove(KSComponent* aParentComponent, XParentType* aParent, KSComponent* aChildComponent,
                          XChildType* aChild, void (XParentType::*anAddMember)(XChildType*),
                          void (XParentType::*aRemoveMember)(XChildType*)) :
        KSCommand(),
        fParentPointer(aParent),
        fChildPointer(aChild),
        fAddMember(anAddMember),
        fRemoveMember(aRemoveMember)
    {
        Set(this);
        fParentComponent = aParentComponent;
        fChildComponent = aChildComponent;
    }
    KSCommandMemberRemove(const KSCommandMemberRemove<XParentType, XChildType>& aCopy) :
        KSCommand(aCopy),
        fParentPointer(aCopy.fParentPointer),
        fChildPointer(aCopy.fChildPointer),
        fAddMember(aCopy.fAddMember),
        fRemoveMember(aCopy.fRemoveMember)

    {
        Set(this);
        fParentComponent = aCopy.fParentComponent;
        fChildComponent = aCopy.fChildComponent;
    }
    ~KSCommandMemberRemove() override {}

  public:
    KSCommandMemberRemove* Clone() const override
    {
        return new KSCommandMemberRemove<XParentType, XChildType>(*this);
    }

  protected:
    void ActivateCommand() override
    {
        fChildComponent->Deactivate();
        (fParentPointer->*fRemoveMember)(fChildPointer);
        return;
    }
    void DeactivateCommand() override
    {
        (fParentPointer->*fAddMember)(fChildPointer);
        fChildComponent->Activate();
        return;
    }

  private:
    XParentType* fParentPointer;
    XChildType* fChildPointer;
    void (XParentType::*fAddMember)(XChildType*);
    void (XParentType::*fRemoveMember)(XChildType*);
};

template<class XParentType, class XChildType> class KSCommandMemberRemoveFactory : public KSCommandFactory
{
  public:
    KSCommandMemberRemoveFactory(void (XParentType::*anAddMember)(XChildType*),
                                 void (XParentType::*aRemoveMember)(XChildType*)) :
        fAddMember(anAddMember),
        fRemoveMember(aRemoveMember)
    {}
    ~KSCommandMemberRemoveFactory() override {}

  public:
    KSCommand* CreateCommand(KSComponent* aParent, KSComponent* aChild) const override
    {
        auto* tParent = aParent->As<XParentType>();
        if (tParent == nullptr) {
            objctmsg_debug("  command parent <" << aParent->GetName() << "> could not be cast to type <"
                                                << typeid(XParentType).name() << ">" << eom);
            return nullptr;
        }

        auto* tChild = aChild->As<XChildType>();
        if (tChild == nullptr) {
            objctmsg_debug("  command child <" << aChild->GetName() << "> could not be cast to type <"
                                               << typeid(XChildType).name() << ">" << eom);
            return nullptr;
        }

        objctmsg_debug("  command built" << eom);
        return new KSCommandMemberRemove<XParentType, XChildType>(aParent,
                                                                  tParent,
                                                                  aChild,
                                                                  tChild,
                                                                  fAddMember,
                                                                  fRemoveMember);
    }

  private:
    void (XParentType::*fAddMember)(XChildType*);
    void (XParentType::*fRemoveMember)(XChildType*);
};


template<class XParentType, class XChildType> class KSCommandMemberParameter : public KSCommand
{
  public:
    KSCommandMemberParameter(KSComponent* aParentComponent, XParentType* aParentPointer, KSComponent* aChildComponent,
                             XChildType* aChildPointer, void (XParentType::*aSetMember)(const XChildType&),
                             const XChildType& (XParentType::*aGetMember)() const) :
        KSCommand(),
        fParentPointer(aParentPointer),
        fChildPointer(aChildPointer),
        fSetMember(aSetMember),
        fGetMember(aGetMember)
    {
        Set(this);
        fParentComponent = aParentComponent;
        fChildComponent = aChildComponent;
    }
    KSCommandMemberParameter(const KSCommandMemberParameter<XParentType, XChildType>& aCopy) :
        KSCommand(aCopy),
        fParentPointer(aCopy.fParentPointer),
        fChildPointer(aCopy.fChildPointer),
        fSetMember(aCopy.fSetMember),
        fGetMember(aCopy.fGetMember)
    {
        Set(this);
        fParentComponent = aCopy.fParentComponent;
        fChildComponent = aCopy.fChildComponent;
    }
    ~KSCommandMemberParameter() override {}

  public:
    KSCommandMemberParameter* Clone() const override
    {
        return new KSCommandMemberParameter<XParentType, XChildType>(*this);
    }

  protected:
    void ActivateCommand() override
    {
        XChildType tOldValue = (fParentPointer->*fGetMember)();
        (fParentPointer->fSetMember)(*fChildPointer);
        (*fChildPointer) = tOldValue;
        return;
    }
    void DeactivateCommand() override
    {
        XChildType tOldValue = (fParentPointer->*fGetMember)();
        (fParentPointer->fSetMember)(*fChildPointer);
        (*fChildPointer) = tOldValue;
        return;
    }

  private:
    XParentType* fParentPointer;
    XChildType* fChildPointer;
    void (XParentType::*fSetMember)(const XChildType&);
    const XChildType& (XParentType::*fGetMember)() const;
};

template<class XParentType, class XChildType> class KSCommandMemberParameterFactory : public KSCommandFactory
{
  public:
    KSCommandMemberParameterFactory(void (XParentType::*aSetMember)(const XChildType&),
                                    const XChildType& (XParentType::*aGetMember)() const) :
        fSetMember(aSetMember),
        fGetMember(aGetMember)
    {}
    ~KSCommandMemberParameterFactory() override {}

  public:
    KSCommand* CreateCommand(KSComponent* aParent, KSComponent* aChild) const override
    {
        XParentType* tParent = aParent->As<XParentType>();
        if (tParent == NULL) {
            objctmsg_debug("  command parent <" << aParent->GetName() << "> could not be cast to type <"
                                                << typeid(XParentType).name() << ">" << eom);
            return nullptr;
        }

        XChildType* tChild = aChild->As<XChildType>();
        if (tChild == NULL) {
            objctmsg_debug("  command child <" << aParent->GetName() << "> could not be cast to type <"
                                               << typeid(XChildType).name() << ">" << eom);
            return nullptr;
        }

        objctmsg_debug("  command built" << eom);
        return new KSCommandMemberParameter<XParentType, XChildType>(aParent,
                                                                     tParent,
                                                                     aChild,
                                                                     tChild,
                                                                     fSetMember,
                                                                     fGetMember);
    }

  private:
    void (XParentType::*fSetMember)(const XChildType&);
    const XChildType& (XParentType::*fGetMember)() const;
};


template<class XType>
template<class XParentType, class XChildType>
int KSDictionary<XType>::AddCommand(void (XParentType::*anAddMember)(XChildType*),
                                    void (XParentType::*aRemoveMember)(XChildType*), const std::string& anAddField,
                                    const std::string& aRemoveField)
{
    if (fCommandFactories == nullptr) {
        fCommandFactories = new CommandFactoryMap();
    }

    auto* tAddFactory = new KSCommandMemberAddFactory<XParentType, XChildType>(anAddMember, aRemoveMember);
    fCommandFactories->insert(CommandFactoryEntry(anAddField, tAddFactory));

    auto* tRemoveFactory = new KSCommandMemberRemoveFactory<XParentType, XChildType>(anAddMember, aRemoveMember);
    fCommandFactories->insert(CommandFactoryEntry(aRemoveField, tRemoveFactory));

    return 0;
}

template<class XType>
template<class XParentType, class XChildType>
int KSDictionary<XType>::AddCommand(void (XParentType::*aSetMember)(const XChildType&),
                                    const XChildType& (XParentType::*aGetMember)() const,
                                    const std::string& aParameterField)
{
    if (fCommandFactories == nullptr) {
        fCommandFactories = new CommandFactoryMap();
    }

    auto* tParameterFactory = new KSCommandMemberParameterFactory<XParentType, XChildType>(aSetMember, aGetMember);
    fCommandFactories->insert(CommandFactoryEntry(aParameterField, tParameterFactory));

    return 0;
}

}  // namespace Kassiopeia

#endif
