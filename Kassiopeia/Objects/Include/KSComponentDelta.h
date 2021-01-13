#ifndef Kassiopeia_KSComponentDelta_h_
#define Kassiopeia_KSComponentDelta_h_

#include "KSComponentValue.h"
#include "KSDictionary.h"
#include "KSNumerical.h"

namespace Kassiopeia
{

template<class XValueType> class KSComponentDelta : public KSComponent
{
  public:
    KSComponentDelta(KSComponent* aParentComponent, XValueType* aParentPointer) : KSComponent(), fDelta(aParentPointer)
    {
        Set(&fDelta);
        this->SetParent(aParentComponent);
        aParentComponent->AddChild(this);
    }
    KSComponentDelta(const KSComponentDelta<XValueType>& aCopy) : KSComponent(aCopy), fDelta(aCopy.fDelta)
    {
        Set(&fDelta);
        this->SetParent(aCopy.fParentComponent);
        aCopy.fParentComponent->AddChild(this);
    }
    ~KSComponentDelta() override = default;

    //***********
    //KSComponent
    //***********

  public:
    KSComponent* Clone() const override
    {
        return new KSComponentDelta<XValueType>(*this);
    }
    KSComponent* Component(const std::string& aField) override
    {
        objctmsg_debug("component delta <" << this->GetName() << "> building component named <" << aField << ">" << eom)
            KSComponent* tComponent = KSDictionary<XValueType>::GetComponent(this, aField);
        if (tComponent == nullptr) {
            objctmsg(eError) << "component delta <" << this->GetName() << "> has no output named <" << aField << ">"
                             << eom;
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
    void InitializeComponent() override
    {
        fDelta.Reset();
    }

    void PushUpdateComponent() override
    {
        objctmsg_debug("component delta <" << this->GetName() << "> pushing update" << eom);
        (void) fDelta.Update();
        return;
    }

    void PullDeupdateComponent() override
    {
        objctmsg_debug("component delta <" << this->GetName() << "> pulling deupdate" << eom);
        fDelta.Reset();
        return;
    }

  private:
    KSComponentValueDelta<XValueType> fDelta;
};

}  // namespace Kassiopeia

#endif
