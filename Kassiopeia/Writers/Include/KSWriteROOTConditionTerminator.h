//
// Created by trost on 07.03.16.
//

#ifndef KASPER_KSWRITEROOTCONDITIONTERMINATOR_H
#define KASPER_KSWRITEROOTCONDITIONTERMINATOR_H

#include "KField.h"
#include "KSComponent.h"
#include "KSWriteROOTCondition.h"

namespace Kassiopeia
{


class KSWriteROOTConditionTerminator : public KSComponentTemplate<KSWriteROOTConditionTerminator, KSWriteROOTCondition>
{
  public:
    KSWriteROOTConditionTerminator() : fComponent(nullptr), fValue(nullptr), fMatchTerminator(std::string("")) {}
    KSWriteROOTConditionTerminator(const KSWriteROOTConditionTerminator& aCopy) :
        KSComponent(aCopy),
        fComponent(aCopy.fComponent),
        fValue(aCopy.fValue),
        fMatchTerminator(aCopy.fMatchTerminator)
    {}
    KSWriteROOTConditionTerminator* Clone() const override
    {
        return new KSWriteROOTConditionTerminator(*this);
    }
    ~KSWriteROOTConditionTerminator() override {}

    void CalculateWriteCondition(bool& aFlag) override
    {
        aFlag = (fValue->compare(fMatchTerminator) == 0);

        return;
    }


  protected:
    ;
    K_SET_GET_PTR(KSComponent, Component);
    ;
    K_SET_GET_PTR(std::string, Value);
    ;
    K_SET_GET(std::string, MatchTerminator);
};

}  // namespace Kassiopeia


#endif  //KASPER_KSWRITEROOTCONDITIONTERMINATOR_H
