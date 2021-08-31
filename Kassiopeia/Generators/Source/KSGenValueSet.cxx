#include "KSGenValueSet.h"

#include "KSGeneratorsMessage.h"

namespace Kassiopeia
{

KSGenValueSet::KSGenValueSet() : fValueStart(0.), fValueStop(0.), fValueIncrement(0.), fValueCount(0) {}
KSGenValueSet::KSGenValueSet(const KSGenValueSet& aCopy) :
    KSComponent(aCopy),
    fValueStart(aCopy.fValueStart),
    fValueStop(aCopy.fValueStop),
    fValueIncrement(aCopy.fValueIncrement),
    fValueCount(aCopy.fValueCount)
{}
KSGenValueSet* KSGenValueSet::Clone() const
{
    return new KSGenValueSet(*this);
}
KSGenValueSet::~KSGenValueSet() = default;

void KSGenValueSet::DiceValue(std::vector<double>& aDicedValues)
{
    double tValue;
    double tValueCount = fValueCount;
    double tValueIncrement = fValueIncrement;

    if (tValueCount != 0)
        tValueIncrement = (fValueStop - fValueStart) / ((double) (tValueCount > 1 ? tValueCount - 1 : 1));

    if (fValueIncrement != 0.) {
        if ((fValueCount > 0) && (fValueIncrement != tValueIncrement))  // only fail if the two definitions do not match
        {
            genmsg(eError) << "generator <" << GetName() << "> cannot dice <" << fValueCount
                           << "> values with a step size of <" << fValueIncrement << ">" << eom;
            return;
        }
        tValueIncrement = fValueIncrement;
        tValueCount = (fValueStop - fValueStart) / tValueIncrement + 1;  // last step included
    }

    for (unsigned int tIndex = 0; tIndex < tValueCount; tIndex++) {
        tValue = fValueStart + tIndex * tValueIncrement;
        aDicedValues.push_back(tValue);
    }
}

}  // namespace Kassiopeia
