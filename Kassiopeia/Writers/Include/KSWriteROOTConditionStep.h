#ifndef Kassiopeia_KSWriteROOTConditionStep_h_
#define Kassiopeia_KSWriteROOTConditionStep_h_

#include "KField.h"
#include "KSComponent.h"
#include "KSWriteROOTCondition.h"

#include <limits>

namespace Kassiopeia
{

template<class XValueType>
class KSWriteROOTConditionStep : public KSComponentTemplate<KSWriteROOTConditionStep<XValueType>, KSWriteROOTCondition>
{
  public:
    KSWriteROOTConditionStep() : fNthStepValue(1), fComponent(nullptr), fValue(nullptr) {}
    KSWriteROOTConditionStep(const KSWriteROOTConditionStep& aCopy) :
        KSComponent(aCopy),
        fNthStepValue(aCopy.fNthStepValue),
        fComponent(aCopy.fComponent),
        fValue(aCopy.fValue)
    {}
    KSWriteROOTConditionStep* Clone() const override
    {
        return new KSWriteROOTConditionStep(*this);
    }
    ~KSWriteROOTConditionStep() override = default;

    void CalculateWriteCondition(bool& aFlag) override
    {
        if (*fValue % fNthStepValue == 0) {
            aFlag = true;
            return;
        }

        aFlag = false;
        return;
    }


  protected:
    ;
    K_SET_GET(int, NthStepValue);
    ;
    K_SET_GET_PTR(KSComponent, Component);
    ;
    K_SET_GET_PTR(XValueType, Value);
};

}  // namespace Kassiopeia

#endif
