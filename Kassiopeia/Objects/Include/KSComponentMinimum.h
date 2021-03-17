#ifndef Kassiopeia_KSComponentMinimum_h_
#define Kassiopeia_KSComponentMinimum_h_

#include "KSComponentValue.h"
#include "KSDictionary.h"
#include "KSNumerical.h"

namespace Kassiopeia
{

template<class XValueType> class KSComponentMinimum : public KSComponent
{
  public:
    KSComponentMinimum(KSComponent* aParentComponent, XValueType* aParentPointer) :
        KSComponent(),
        fMinimum(aParentPointer)
    {
        Set(&fMinimum);
        this->SetParent(aParentComponent);
        aParentComponent->AddChild(this);
    }
    KSComponentMinimum(const KSComponentMinimum<XValueType>& aCopy) : KSComponent(aCopy), fMinimum(aCopy.fMinimum)
    {
        Set(&fMinimum);
        this->SetParent(aCopy.fParentComponent);
        aCopy.fParentComponent->AddChild(this);
    }
    ~KSComponentMinimum() override = default;

    //***********
    //KSComponent
    //***********

  public:
    KSComponent* Clone() const override
    {
        return new KSComponentMinimum<XValueType>(*this);
    }
    KSComponent* Component(const std::string& aField) override
    {
        objctmsg_debug("component minimum <" << this->GetName() << "> building component named <" << aField << ">"
                                             << eom) KSComponent* tComponent =
            KSDictionary<XValueType>::GetComponent(this, aField);
        if (tComponent == nullptr) {
            objctmsg(eError) << "component minimum <" << this->GetName() << "> has no output named <" << aField << ">"
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
        fMinimum.Reset();
    }

    void PushUpdateComponent() override
    {
        objctmsg_debug("component minimum <" << this->GetName() << "> pushing update" << eom);
        (void) fMinimum.Update();
    }

    void PullDeupdateComponent() override
    {
        objctmsg_debug("component minimum <" << this->GetName() << "> pulling deupdate" << eom);
        fMinimum.Reset();
    }

  private:
    KSComponentValueMinimum<XValueType> fMinimum;
};

}  // namespace Kassiopeia

#endif
