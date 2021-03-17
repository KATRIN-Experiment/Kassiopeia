#ifndef Kassiopeia_KSWriteROOTConditionOutput_h_
#define Kassiopeia_KSWriteROOTConditionOutput_h_

#include "KField.h"
#include "KSComponent.h"
#include "KSWriteROOTCondition.h"

#include <limits>

namespace Kassiopeia
{

template<class XValueType>
class KSWriteROOTConditionOutput :
    public KSComponentTemplate<KSWriteROOTConditionOutput<XValueType>, KSWriteROOTCondition>
{
  public:
    KSWriteROOTConditionOutput() :
        fMinValue(std::numeric_limits<XValueType>::lowest()),
        fMaxValue(std::numeric_limits<XValueType>::max()),
        fComponent(nullptr),
        fValue(nullptr)
    {}
    KSWriteROOTConditionOutput(const KSWriteROOTConditionOutput& aCopy) :
        KSComponent(aCopy),
        fMinValue(aCopy.fMinValue),
        fMaxValue(aCopy.fMaxValue),
        fComponent(aCopy.fComponent),
        fValue(aCopy.fValue)
    {}
    KSWriteROOTConditionOutput* Clone() const override
    {
        return new KSWriteROOTConditionOutput(*this);
    }
    ~KSWriteROOTConditionOutput() override = default;

    void CalculateWriteCondition(bool& aFlag) override
    {
        if (*fValue >= fMaxValue || *fValue <= fMinValue) {
            aFlag = false;
            return;
        }

        aFlag = true;
        return;
    }


  protected:
    ;
    K_SET_GET(XValueType, MinValue);
    ;
    K_SET_GET(XValueType, MaxValue);
    ;
    K_SET_GET_PTR(KSComponent, Component);
    ;
    K_SET_GET_PTR(XValueType, Value);
};

}  // namespace Kassiopeia

#endif
