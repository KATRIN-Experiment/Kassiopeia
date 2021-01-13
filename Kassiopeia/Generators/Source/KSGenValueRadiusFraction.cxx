#include "KSGenValueRadiusFraction.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"
using katrin::KRandom;

#include "KConst.h"

namespace Kassiopeia
{

KSGenValueRadiusFraction::KSGenValueRadiusFraction() = default;

KSGenValueRadiusFraction::KSGenValueRadiusFraction(const KSGenValueRadiusFraction& /*aCopy*/) : KSComponent() {}
KSGenValueRadiusFraction* KSGenValueRadiusFraction::Clone() const
{
    return new KSGenValueRadiusFraction(*this);
}
KSGenValueRadiusFraction::~KSGenValueRadiusFraction() = default;

void KSGenValueRadiusFraction::DiceValue(std::vector<double>& aDicedValues)
{

    double tRadiusF = std::pow(KRandom::GetInstance().Uniform(0., 1.), (1. / 2.));

    aDicedValues.push_back(tRadiusF);

    return;
}

}  // namespace Kassiopeia
