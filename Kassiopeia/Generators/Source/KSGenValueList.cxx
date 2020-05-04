#include "KSGenValueList.h"

namespace Kassiopeia
{

KSGenValueList::KSGenValueList() : fValues() {}
KSGenValueList::KSGenValueList(const KSGenValueList& aCopy) : KSComponent(), fValues(aCopy.fValues) {}
KSGenValueList* KSGenValueList::Clone() const
{
    return new KSGenValueList(*this);
}
KSGenValueList::~KSGenValueList() {}

void KSGenValueList::DiceValue(vector<double>& aDicedValues)
{
    aDicedValues = fValues;
}

void KSGenValueList::AddValue(double aValue)
{
    fValues.push_back(aValue);
}


}  // namespace Kassiopeia
