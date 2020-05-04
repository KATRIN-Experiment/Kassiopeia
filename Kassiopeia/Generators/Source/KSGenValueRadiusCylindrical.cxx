#include "KSGenValueRadiusCylindrical.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"
using katrin::KRandom;

#include "KConst.h"

namespace Kassiopeia
{

KSGenValueRadiusCylindrical::KSGenValueRadiusCylindrical() : fRadiusMin(0.), fRadiusMax(0.) {}
KSGenValueRadiusCylindrical::KSGenValueRadiusCylindrical(const KSGenValueRadiusCylindrical& aCopy) :
    KSComponent(),
    fRadiusMin(aCopy.fRadiusMin),
    fRadiusMax(aCopy.fRadiusMax)
{}
KSGenValueRadiusCylindrical* KSGenValueRadiusCylindrical::Clone() const
{
    return new KSGenValueRadiusCylindrical(*this);
}
KSGenValueRadiusCylindrical::~KSGenValueRadiusCylindrical() {}

void KSGenValueRadiusCylindrical::DiceValue(vector<double>& aDicedValues)
{
    double tMinRadiusSquared = fRadiusMin * fRadiusMin;
    double tMaxRadiusSquared = fRadiusMax * fRadiusMax;
    double tRadius = pow(KRandom::GetInstance().Uniform(tMinRadiusSquared, tMaxRadiusSquared), (1. / 2.));

    aDicedValues.push_back(tRadius);

    return;
}

}  // namespace Kassiopeia
