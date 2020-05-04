#include "KSGenStringValueFix.h"

namespace Kassiopeia
{

KSGenStringValueFix::KSGenStringValueFix() : fValue("") {}
KSGenStringValueFix::KSGenStringValueFix(const KSGenStringValueFix& aCopy) : KSComponent(), fValue(aCopy.fValue) {}
KSGenStringValueFix* KSGenStringValueFix::Clone() const
{
    return new KSGenStringValueFix(*this);
}
KSGenStringValueFix::~KSGenStringValueFix() {}

void KSGenStringValueFix::DiceValue(vector<std::string>& aDicedValues)
{
    std::string tValue = fValue;
    aDicedValues.push_back(tValue);
}

}  // namespace Kassiopeia
