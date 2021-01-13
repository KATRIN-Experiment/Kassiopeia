#include "KSGenValueList.h"

#include "KRandom.h"
using katrin::KRandom;

namespace Kassiopeia
{

KSGenValueList::KSGenValueList() : fValues(), fRandomize(false) {}
KSGenValueList::KSGenValueList(const KSGenValueList& aCopy) :
    KSComponent(aCopy),
    fValues(aCopy.fValues),
    fRandomize(aCopy.fRandomize)
{}
KSGenValueList* KSGenValueList::Clone() const
{
    return new KSGenValueList(*this);
}
KSGenValueList::~KSGenValueList() = default;

void KSGenValueList::DiceValue(std::vector<double>& aDicedValues)
{
    if (fRandomize) {
        auto tIndex = KRandom::GetInstance().Uniform(0UL, fValues.size() - 1);
        double tValue = fValues.at(tIndex);
        aDicedValues.push_back(tValue);
        return;
    }

    aDicedValues = fValues;
}

void KSGenValueList::AddValue(double aValue)
{
    fValues.push_back(aValue);
}

void KSGenValueList::SetRandomize(bool aFlag)
{
    fRandomize = aFlag;
}

}  // namespace Kassiopeia
