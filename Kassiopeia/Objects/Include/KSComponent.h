#ifndef Kassiopeia_KSComponent_h_
#define Kassiopeia_KSComponent_h_

#include "KSObject.h"

namespace Kassiopeia
{

class KSCommand;

class KSComponent : public KSObject
{
  public:
    KSComponent();
    KSComponent(const KSComponent& aCopy);
    ~KSComponent() override;

  public:
    KSComponent* Clone() const override = 0;
    virtual KSComponent* Component(const std::string& aField) = 0;
    virtual KSCommand* Command(const std::string& aField, KSComponent* aChild) = 0;

  public:
    using StateType = enum
    {
        eIdle = 0,
        eInitialized = 1,
        eActivated = 2,
        eUpdated = 3
    };

    const StateType& State() const;
    bool IsInitialized() const;
    bool IsActivated() const;

    void TryInitialize();
    void Initialize();
    void Deinitialize();
    void TryActivate();
    void Activate();
    void Deactivate();
    void PushUpdate();
    void PushDeupdate();
    void PullUpdate();
    void PullDeupdate();

  protected:
    StateType fState;

    virtual void InitializeComponent();
    virtual void DeinitializeComponent();
    virtual void ActivateComponent();
    virtual void DeactivateComponent();
    virtual void PushUpdateComponent();
    virtual void PushDeupdateComponent();
    virtual void PullUpdateComponent();
    virtual void PullDeupdateComponent();

  public:
    void SetParent(KSComponent* aParent);
    KSComponent* GetParent() const;

    void AddChild(KSComponent* aChild);
    unsigned int GetChildCount() const;
    KSComponent* GetChild(const unsigned int& anIndex) const;

  protected:
    KSComponent* fParentComponent;
    std::vector<KSComponent*> fChildComponents;
};

inline const KSComponent::StateType& KSComponent::State() const
{
    return fState;
}

inline bool KSComponent::IsInitialized() const
{
    return (fState == eInitialized) || (fState == eActivated);
}

inline bool KSComponent::IsActivated() const
{
    return (fState == eActivated);
}

template<> inline bool KSObject::Is<KSComponent>()
{
    auto* tComponent = dynamic_cast<KSComponent*>(this);
    if (tComponent != nullptr) {
        return true;
    }
    return false;
}

template<> inline bool KSObject::Is<KSComponent>() const
{
    const auto* tComponent = dynamic_cast<const KSComponent*>(this);
    if (tComponent != nullptr) {
        return true;
    }
    return false;
}

template<> inline KSComponent* KSObject::As<KSComponent>()
{
    auto* tComponent = dynamic_cast<KSComponent*>(this);
    if (tComponent != nullptr) {
        return tComponent;
    }
    return nullptr;
}

template<> inline const KSComponent* KSObject::As<KSComponent>() const
{
    const auto* tComponent = dynamic_cast<const KSComponent*>(this);
    if (tComponent != nullptr) {
        return tComponent;
    }
    return nullptr;
}

}  // namespace Kassiopeia

#endif
