#ifndef Kassiopeia_KSComponentMinimumAt_h_
#define Kassiopeia_KSComponentMinimumAt_h_

#include "KSComponentValue.h"
#include "KSDictionary.h"
#include "KSNumerical.h"

namespace Kassiopeia
{

template<class XValueType, class XValueTypeSource> class KSComponentMinimumAt : public KSComponent
{
  public:
    KSComponentMinimumAt(KSComponent* aParentComponent, XValueType* aParentPointer, XValueTypeSource* aSourcePointer) :
        KSComponent(),
        fMinimum(aSourcePointer),
        fOperand(aParentPointer),
        fMinimumAt(KSNumerical<XValueType>::Zero())
    {
        Set(&fMinimumAt);
        this->SetParent(aParentComponent);
        aParentComponent->AddChild(this);
    }
    KSComponentMinimumAt(const KSComponentMinimumAt<XValueType, XValueTypeSource>& aCopy) :
        KSComponent(aCopy),
        fMinimum(aCopy.fMinimum),
        fOperand(aCopy.fOperand),
        fMinimumAt(aCopy.fMinimumAt)
    {
        Set(&fMinimumAt);
        this->SetParent(aCopy.fParentComponent);
        aCopy.fParentComponent->AddChild(this);
    }
    ~KSComponentMinimumAt() override {}

    //***********
    //KSComponent
    //***********

  public:
    KSComponent* Clone() const override
    {
        return new KSComponentMinimumAt<XValueType, XValueTypeSource>(*this);
    }
    KSComponent* Component(const std::string& aField) override
    {
        objctmsg_debug("component minimum_at <" << this->GetName() << "> building component named <" << aField << ">"
                                                << eom) KSComponent* tComponent =
            KSDictionary<XValueType>::GetComponent(this, aField);
        if (tComponent == nullptr) {
            objctmsg(eError) << "component minimum_at <" << this->GetName() << "> has no output named <" << aField
                             << ">" << eom;
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
        fMinimum.Reset();
    }

    void PushUpdateComponent() override
    {
        objctmsg_debug("component minimum_at <" << this->GetName() << "> pushing update" << eom);
        if (fMinimum.Update() == true) {
            fMinimumAt = (*fOperand);
        }
    }

    void PullDeupdateComponent() override
    {
        objctmsg_debug("component minimum_at <" << this->GetName() << "> pulling deupdate" << eom);
        fMinimum.Reset();
    }

  private:
    KSComponentValueMinimum<XValueTypeSource> fMinimum;
    XValueType* fOperand;
    XValueType fMinimumAt;
};

}  // namespace Kassiopeia

#endif
