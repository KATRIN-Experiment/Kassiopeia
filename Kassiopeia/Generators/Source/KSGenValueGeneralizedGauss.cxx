#include "KSGenValueGeneralizedGauss.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"
using katrin::KRandom;

namespace Kassiopeia
{

KSGenValueGeneralizedGauss::KSGenValueGeneralizedGauss() :
    fValueMin(0.),
    fValueMax(0.),
    fValueMean(0.),
    fValueSigma(0.),
    fValueSkew(0.),
    fSolver()
{}
KSGenValueGeneralizedGauss::KSGenValueGeneralizedGauss(const KSGenValueGeneralizedGauss& aCopy) :
    KSComponent(),
    fValueMin(aCopy.fValueMin),
    fValueMax(aCopy.fValueMax),
    fValueMean(aCopy.fValueMean),
    fValueSigma(aCopy.fValueSigma),
    fValueSkew(aCopy.fValueSkew),
    fSolver()
{}
KSGenValueGeneralizedGauss* KSGenValueGeneralizedGauss::Clone() const
{
    return new KSGenValueGeneralizedGauss(*this);
}
KSGenValueGeneralizedGauss::~KSGenValueGeneralizedGauss() {}

void KSGenValueGeneralizedGauss::DiceValue(vector<double>& aDicedValues)
{
    double tValue;

    if (fValueMin == fValueMax) {
        double tValueGauss = KRandom::GetInstance().Uniform(0., 1.);
        fSolver.Solve(KMathBracketingSolver::eBrent,
                      this,
                      &KSGenValueGeneralizedGauss::ValueFunction,
                      tValueGauss,
                      -1e20,
                      1e20,
                      tValue);  // fake (-inf,inf) integration by using (-1e20,1e20)
    }
    else {
        double tValueGaussMin = ValueFunction(fValueMin);
        double tValueGaussMax = ValueFunction(fValueMax);
        double tValueGauss = KRandom::GetInstance().Uniform(tValueGaussMin, tValueGaussMax);
        fSolver.Solve(KMathBracketingSolver::eBrent,
                      this,
                      &KSGenValueGeneralizedGauss::ValueFunction,
                      tValueGauss,
                      fValueMin,
                      fValueMax,
                      tValue);
    }

    aDicedValues.push_back(tValue);

    return;
}

double KSGenValueGeneralizedGauss::ValueFunction(const double& aValue) const
{
    // CDF of generalized normal distribution (version 2)
    if (fValueSkew == 0.) {
        return .5 * (1. + erf((aValue - fValueMean) / (sqrt(2.) * fValueSigma)));
    }

    double lim = fValueMean + fValueSigma / fValueSkew;
    if ((fValueSkew > 0.) && (aValue >= lim))
        return 1.;
    else if ((fValueSkew < 0.) && (aValue <= lim))
        return 0.;

    double tExp = -1. / fValueSkew * log(1. - fValueSkew * (aValue - fValueMean) / fValueSigma);
    return .5 * (1. + erf(tExp / sqrt(2.)));
}

}  // namespace Kassiopeia
