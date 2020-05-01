#include "KSGenValueGauss.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"
using katrin::KRandom;

namespace Kassiopeia
{

KSGenValueGauss::KSGenValueGauss() : fValueMin(0.), fValueMax(0.), fValueMean(0.), fValueSigma(0.), fSolver() {}
KSGenValueGauss::KSGenValueGauss(const KSGenValueGauss& aCopy) :
    KSComponent(),
    fValueMin(aCopy.fValueMin),
    fValueMax(aCopy.fValueMax),
    fValueMean(aCopy.fValueMean),
    fValueSigma(aCopy.fValueSigma),
    fSolver()
{}
KSGenValueGauss* KSGenValueGauss::Clone() const
{
    return new KSGenValueGauss(*this);
}
KSGenValueGauss::~KSGenValueGauss() {}

void KSGenValueGauss::DiceValue(vector<double>& aDicedValues)
{
    double tValue;

    if (fValueMin == fValueMax) {
        tValue = KRandom::GetInstance().Gauss(fValueMean, fValueSigma);
    }
    else {
        double tValueGaussMin = ValueFunction(fValueMin);
        double tValueGaussMax = ValueFunction(fValueMax);
        double tValueGauss = KRandom::GetInstance().Uniform(tValueGaussMin, tValueGaussMax);
        fSolver.Solve(KMathBracketingSolver::eBrent,
                      this,
                      &KSGenValueGauss::ValueFunction,
                      tValueGauss,
                      fValueMin,
                      fValueMax,
                      tValue);
    }

    aDicedValues.push_back(tValue);

    return;
}

double KSGenValueGauss::ValueFunction(const double& aValue) const
{
    return .5 * (1. + erf((aValue - fValueMean) / (sqrt(2.) * fValueSigma)));
}

}  // namespace Kassiopeia
