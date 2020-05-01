#include "KSGenValuePareto.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"
using katrin::KRandom;

namespace Kassiopeia
{

KSGenValuePareto::KSGenValuePareto() : fSlope(1.), fCutoff(1.), fOffset(0.), fValueMin(-1.), fValueMax(-1.) {}
KSGenValuePareto::KSGenValuePareto(const KSGenValuePareto& aCopy) :
    KSComponent(),
    fSlope(aCopy.fSlope),
    fCutoff(aCopy.fCutoff),
    fOffset(aCopy.fOffset),
    fValueMin(aCopy.fValueMin),
    fValueMax(aCopy.fValueMax)
{}
KSGenValuePareto* KSGenValuePareto::Clone() const
{
    return new KSGenValuePareto(*this);
}
KSGenValuePareto::~KSGenValuePareto() {}

void KSGenValuePareto::DiceValue(vector<double>& aDicedValues)
{
    double tValue;

    if (fValueMin == fValueMax) {
        tValue = fOffset + fCutoff * std::exp(KRandom::GetInstance().Exponential(fSlope));
    }
    else {
        double tUniform = fValueParetoMax +
                          KRandom::GetInstance().Uniform(0.0, 1.0, false, true) * (fValueParetoMin - fValueParetoMax);
        tValue = fOffset + fCutoff * std::exp((-log(tUniform) * fSlope));
    }

    aDicedValues.push_back(tValue);

    return;
}

void KSGenValuePareto::InitializeComponent()
{
    if (fValueMin != -1.)
        fValueParetoMin = std::exp(-log((fValueMin - fOffset) / fCutoff) / fSlope);
    if (fValueMax != -1.)
        fValueParetoMax = std::exp(-log((fValueMax - fOffset) / fCutoff) / fSlope);

    return;
}
void KSGenValuePareto::DeinitializeComponent()
{
    return;
}

}  // namespace Kassiopeia
