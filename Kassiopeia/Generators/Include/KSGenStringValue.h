#ifndef _Kassiopeia_KSGenStringValue_h_
#define _Kassiopeia_KSGenStringValue_h_

#include "KSComponentTemplate.h"

#include <vector>

namespace Kassiopeia
{

class KSGenStringValue : public KSComponentTemplate<KSGenStringValue>
{
  public:
    KSGenStringValue();
    ~KSGenStringValue() override;

  public:
    virtual void DiceValue(std::vector<std::string>& aDicedValue) = 0;
};

}  // namespace Kassiopeia

#endif
