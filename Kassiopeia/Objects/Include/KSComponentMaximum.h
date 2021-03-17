#ifndef Kassiopeia_KSComponentMaximum_h_
#define Kassiopeia_KSComponentMaximum_h_

#include "KSComponentValue.h"
#include "KSDictionary.h"
#include "KSNumerical.h"

namespace Kassiopeia
{

template<class XValueType> class KSComponentMaximum : public KSComponent
{
  public:
    KSComponentMaximum(KSComponent* aParentComponent, XValueType* aParentPointer) :
        KSComponent(),
        fMaximum(aParentPointer)
    {
        Set(&fMaximum);
        this->SetParent(aParentComponent);
        aParentComponent->AddChild(this);
    }
    KSComponentMaximum(const KSComponentMaximum<XValueType>& aCopy) : KSComponent(aCopy), fMaximum(aCopy.fMaximum)
    {
        Set(&fMaximum);
        this->SetParent(aCopy.fParentComponent);
        aCopy.fParentComponent->AddChild(this);
    }
    ~KSComponentMaximum() override = default;

    //***********
    //KSComponent
    //***********

  public:
    KSComponent* Clone() const override
    {
        return new KSComponentMaximum<XValueType>(*this);
    }
    KSComponent* Component(const std::string& aField) override
    {
        objctmsg_debug("component maximum <" << this->GetName() << "> building component named <" << aField << ">"
                                             << eom) KSComponent* tComponent =
            KSDictionary<XValueType>::GetComponent(this, aField);
        if (tComponent == nullptr) {
            objctmsg(eError) << "component maximum <" << this->GetName() << "> has no output named <" << aField << ">"
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
        fMaximum.Reset();
    }

    void PushUpdateComponent() override
    {
        objctmsg_debug("component maximum <" << this->GetName() << "> pushing update" << eom);
        (void) fMaximum.Update();
    }

    void PullDeupdateComponent() override
    {
        objctmsg_debug("component maximum <" << this->GetName() << "> pulling deupdate" << eom);
        fMaximum.Reset();
    }

  private:
    KSComponentValueMaximum<XValueType> fMaximum;
};

}  // namespace Kassiopeia

#endif
