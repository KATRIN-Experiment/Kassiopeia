#ifndef Kassiopeia_KSWriteROOTCondition_h_
#define Kassiopeia_KSWriteROOTCondition_h_

#include "KSComponentTemplate.h"

namespace Kassiopeia
{

class KSWriteROOTCondition : public KSComponentTemplate<KSWriteROOTCondition>
{
  public:
    KSWriteROOTCondition();
    ~KSWriteROOTCondition() override;

  public:
    virtual void CalculateWriteCondition(bool& aFlag) = 0;
};


}  // namespace Kassiopeia

#endif
