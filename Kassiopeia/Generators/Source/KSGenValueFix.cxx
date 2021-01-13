#include "KSGenValueFix.h"

namespace Kassiopeia
{

KSGenValueFix::KSGenValueFix() : fValue(0.) {}
KSGenValueFix::KSGenValueFix(const KSGenValueFix& aCopy) : KSComponent(aCopy), fValue(aCopy.fValue) {}
KSGenValueFix* KSGenValueFix::Clone() const
{
    return new KSGenValueFix(*this);
}
KSGenValueFix::~KSGenValueFix() = default;

void KSGenValueFix::DiceValue(std::vector<double>& aDicedValues)
{
    double tValue = fValue;
    aDicedValues.push_back(tValue);
}

}  // namespace Kassiopeia
