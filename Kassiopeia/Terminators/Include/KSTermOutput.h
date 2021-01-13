#ifndef Kassiopeia_KSTermOutput_h_
#define Kassiopeia_KSTermOutput_h_

#include "KField.h"
#include "KSComponent.h"
#include "KSParticle.h"
#include "KSTerminator.h"

#include <limits>

namespace Kassiopeia
{

template<class XValueType> class KSTermOutput : public KSComponentTemplate<KSTermOutput<XValueType>, KSTerminator>
{
  public:
    KSTermOutput() :
        fMinValue(std::numeric_limits<XValueType>::lowest()),
        fMaxValue(std::numeric_limits<XValueType>::max()),
        fValue(nullptr),
        fFirstStep(true)
    {}
    KSTermOutput(const KSTermOutput& aCopy) :
        KSComponent(aCopy),
        fMinValue(aCopy.fMinValue),
        fMaxValue(aCopy.fMaxValue),
        fValue(aCopy.fValue),
        fFirstStep(aCopy.fFirstStep)
    {}
    KSTermOutput* Clone() const override
    {
        return new KSTermOutput(*this);
    }
    ~KSTermOutput() override = default;

    void CalculateTermination(const KSParticle& /*anInitialParticle*/, bool& aFlag) override
    {
        if (fFirstStep == true) {
            fFirstStep = false;
            aFlag = false;
            return;
        }

        if (*fValue >= fMaxValue || *fValue <= fMinValue) {
            aFlag = true;
            return;
        }

        aFlag = false;
        return;
    }

    void ExecuteTermination(const KSParticle& /*anInitialParticle*/, KSParticle& aFinalParticle,
                            KSParticleQueue& /*aParticleQueue*/) const override
    {
        aFinalParticle.SetActive(false);
        aFinalParticle.SetLabel(katrin::KNamed::GetName());
        return;
    }

    void ActivateComponent() override
    {
        fFirstStep = true;
    }

  protected:
    ;
    K_SET_GET(XValueType, MinValue);
    ;
    K_SET_GET(XValueType, MaxValue);
    ;
    K_SET_GET_PTR(XValueType, Value);
    bool fFirstStep;
};

}  // namespace Kassiopeia

#endif
