#include "KSGenValueZFrustrum.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"
using katrin::KRandom;

#include "KConst.h"

namespace Kassiopeia
{

KSGenValueZFrustrum::KSGenValueZFrustrum() : fr1(0.), fr2(0.), fz1(0.), fz2(0.) {}
KSGenValueZFrustrum::KSGenValueZFrustrum(const KSGenValueZFrustrum& aCopy) :
    KSComponent(),
    fr1(aCopy.fr1),
    fr2(aCopy.fr2),
    fz1(aCopy.fz1),
    fz2(aCopy.fz2)
{}
KSGenValueZFrustrum* KSGenValueZFrustrum::Clone() const
{
    return new KSGenValueZFrustrum(*this);
}
KSGenValueZFrustrum::~KSGenValueZFrustrum() {}

void KSGenValueZFrustrum::DiceValue(vector<double>& aDicedValues)
{
    bool done = false;
    double z = 0;
    double p = 0;

    while (!done) {
        z = KRandom::GetInstance().Uniform(fz1, fz2);
        p = KRandom::GetInstance().Uniform(0., 1.);
        if (p * fr2 * (fz2 - fz1) < (z - fz1) * (fr2 - fr1) + (fz2 - fz1) * fr1) {
            done = true;
        }
    }

    aDicedValues.push_back(z);

    return;
}

}  // namespace Kassiopeia
