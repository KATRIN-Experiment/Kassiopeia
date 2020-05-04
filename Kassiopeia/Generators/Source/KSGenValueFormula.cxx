#include "KSGenValueFormula.h"

namespace Kassiopeia
{

KSGenValueFormula::KSGenValueFormula() : fValueMin(0.), fValueMax(0.), fValueFormula("x"), fValueFunction(nullptr) {}
KSGenValueFormula::KSGenValueFormula(const KSGenValueFormula& aCopy) :
    KSComponent(),
    fValueMin(aCopy.fValueMin),
    fValueMax(aCopy.fValueMax),
    fValueFormula(aCopy.fValueFormula),
    fValueFunction(nullptr)
{}
KSGenValueFormula* KSGenValueFormula::Clone() const
{
    return new KSGenValueFormula(*this);
}
KSGenValueFormula::~KSGenValueFormula() {}

void KSGenValueFormula::DiceValue(vector<double>& aDicedValues)
{
    double tValue;

    tValue = fValueFunction->GetRandom(fValueMin, fValueMax);
    aDicedValues.push_back(tValue);

    return;
}

void KSGenValueFormula::InitializeComponent()
{
    fValueFunction = new TF1("function", fValueFormula.c_str(), fValueMin, fValueMax);
    return;
}
void KSGenValueFormula::DeinitializeComponent()
{
    delete fValueFunction;
    fValueFunction = nullptr;
    return;
}

}  // namespace Kassiopeia
