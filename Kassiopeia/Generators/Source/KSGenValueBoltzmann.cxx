#include "KSGenValueBoltzmann.h"

#include "KConst.h"
#include "KRandom.h"
#include "KSGeneratorsMessage.h"

#include <cmath>
//using katrin::KRandom;
using namespace katrin;

namespace Kassiopeia
{

KSGenValueBoltzmann::KSGenValueBoltzmann() : fValueMass(1.), fValuekT(1.), fUseElectronVolts(false) {}
KSGenValueBoltzmann::KSGenValueBoltzmann(const KSGenValueBoltzmann& aCopy) :
    KSComponent(aCopy),
    fValueMass(aCopy.fValueMass),
    fValuekT(aCopy.fValuekT)
{}
KSGenValueBoltzmann* KSGenValueBoltzmann::Clone() const
{
    return new KSGenValueBoltzmann(*this);
}
KSGenValueBoltzmann::~KSGenValueBoltzmann() = default;

void KSGenValueBoltzmann::DiceValue(std::vector<double>& aDicedValues)
{
    double tValue;

    double fValueSigma = sqrt(fValuekT / fValueMass);
    double v1 = katrin::KRandom::GetInstance().Gauss(0., fValueSigma);
    double v2 = katrin::KRandom::GetInstance().Gauss(0., fValueSigma);
    double v3 = katrin::KRandom::GetInstance().Gauss(0., fValueSigma);
    tValue = 0.5 * fValueMass * (v1 * v1 + v2 * v2 + v3 * v3);
    if (!fUseElectronVolts)
        tValue /= katrin::KConst::Q();

    aDicedValues.push_back(tValue);

    return;
}

}  // namespace Kassiopeia
