#ifndef Kassiopeia_KSComponentIntegral_h_
#define Kassiopeia_KSComponentIntegral_h_

#include "KSComponentValue.h"
#include "KSDictionary.h"
#include "KSNumerical.h"

namespace Kassiopeia
{

template<class XValueType> class KSComponentIntegral : public KSComponent
{
  public:
    KSComponentIntegral(KSComponent* aParentComponent, XValueType* aParentPointer) :
        KSComponent(),
        fIntegral(aParentPointer)
    {
        Set(&fIntegral);
        this->SetParent(aParentComponent);
        aParentComponent->AddChild(this);
    }
    KSComponentIntegral(const KSComponentIntegral<XValueType>& aCopy) : KSComponent(aCopy), fIntegral(aCopy.fIntegral)
    {
        Set(&fIntegral);
        this->SetParent(aCopy.fParentComponent);
        aCopy.fParentComponent->AddChild(this);
    }
    ~KSComponentIntegral() override {}

    //***********
    //KSComponent
    //***********

  public:
    KSComponent* Clone() const override
    {
        return new KSComponentIntegral<XValueType>(*this);
    }
    KSComponent* Component(const std::string& aField) override
    {
        objctmsg_debug("component integral <" << this->GetName() << "> building component named <" << aField << ">"
                                              << eom) KSComponent* tComponent =
            KSDictionary<XValueType>::GetComponent(this, aField);
        if (tComponent == nullptr) {
            objctmsg(eError) << "component integral <" << this->GetName() << "> has no output named <" << aField << ">"
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
        fIntegral.Reset();
    }

    void PushUpdateComponent() override
    {
        objctmsg_debug("component integral <" << this->GetName() << "> pushing update" << eom);
        (void) fIntegral.Update();
        return;
    }

    void PullDeupdateComponent() override
    {
        objctmsg_debug("component integral <" << this->GetName() << "> pulling deupdate" << eom);
        fIntegral.Reset();
        return;
    }

  private:
    KSComponentValueIntegral<XValueType> fIntegral;
};

}  // namespace Kassiopeia

#endif
