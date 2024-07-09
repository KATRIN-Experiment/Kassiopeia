#include "KSGenValueFermi.h"

#include "KRandom.h"
#include "KSGeneratorsMessage.h"

#include "KConst.h"
using katrin::KRandom;

namespace Kassiopeia
{

KSGenValueFermi::KSGenValueFermi() :
    fValueMin(0.),
    fValueMax(0.),
    fValueMean(0.),
    fValueTau(0.),
    fValueTemp(0.),
    fSolver()
{}
KSGenValueFermi::KSGenValueFermi(const KSGenValueFermi& aCopy) :
    KSComponent(aCopy),
    fValueMin(aCopy.fValueMin),
    fValueMax(aCopy.fValueMax),
    fValueMean(aCopy.fValueMean),
    fValueTau(aCopy.fValueTau),
    fValueTemp(aCopy.fValueTemp),
    fSolver()
{}
KSGenValueFermi* KSGenValueFermi::Clone() const
{
    return new KSGenValueFermi(*this);
}
KSGenValueFermi::~KSGenValueFermi() = default;

void KSGenValueFermi::DiceValue(std::vector<double>& aDicedValues)
{
    double tValue;
    double tValueX;
    double tValueY;
    int tCount = 0;
    int tLimit = 100000;
    bool tValueInside = false;

//  Generate Random Values via Von Neumann method https://en.wikipedia.org/wiki/Rejection_sampling
    
    do {
        tValueX = KRandom::GetInstance().Uniform(fValueMin, fValueMax);
        tValueY = KRandom::GetInstance().Uniform(0., 1.); 

        tValue = ValueFunction(tValueX);

        if ((0 <= tValueY) && (tValueY <= tValue)){
            tValueInside = true;
        }

        tCount = tCount + 1;  
        if ((tCount > tLimit) && (tValueInside == false)){
            genmsg(eError) << "Unable to find valid value after " << tLimit << " tries. Shutting down." << ret;
        }

    } while (tValueInside == false);

    aDicedValues.push_back(tValueX);

    return;
}

double KSGenValueFermi::ValueFunction(const double& aValue) const
{
    // Fermi Distribution as seen in J. Behrens PhD (2016) eq. 4.38


    double tSqrt = (1. + fValueTau*aValue - fValueMean) / (1. + fValueMean * (fValueTau - 1.));
    double tExp = (aValue - fValueMean) / (katrin::KConst::kB() * fValueTemp / katrin::KConst::Q());

    return sqrt(tSqrt) * (1. / (exp(tExp) + 1)) ;

}

}  // namespace Kassiopeia
