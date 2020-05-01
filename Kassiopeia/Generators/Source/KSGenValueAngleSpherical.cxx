#include "KSGenValueAngleSpherical.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"
using katrin::KRandom;

#include "KConst.h"

namespace Kassiopeia
{

KSGenValueAngleSpherical::KSGenValueAngleSpherical() : fAngleMin(0.), fAngleMax(0.) {}
KSGenValueAngleSpherical::KSGenValueAngleSpherical(const KSGenValueAngleSpherical& aCopy) :
    KSComponent(),
    fAngleMin(aCopy.fAngleMin),
    fAngleMax(aCopy.fAngleMax)
{}
KSGenValueAngleSpherical* KSGenValueAngleSpherical::Clone() const
{
    return new KSGenValueAngleSpherical(*this);
}
KSGenValueAngleSpherical::~KSGenValueAngleSpherical() {}

void KSGenValueAngleSpherical::DiceValue(vector<double>& aDicedValues)
{
    genmsg_assert(fAngleMax, >= 0);
    genmsg_assert(fAngleMin, >= 0);
    genmsg_assert(fAngleMax, <= 180);
    genmsg_assert(fAngleMin, <= 180);
    double tCosThetaMin = std::cos((katrin::KConst::Pi() / 180.) * fAngleMax);
    double tCosThetaMax = std::cos((katrin::KConst::Pi() / 180.) * fAngleMin);
    double tAngle = std::acos(KRandom::GetInstance().Uniform(tCosThetaMin, tCosThetaMax));

    aDicedValues.push_back((180.0 / katrin::KConst::Pi()) * tAngle);

    return;
}

}  // namespace Kassiopeia
