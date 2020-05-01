#ifndef Kassiopeia_KSMathRK86_h_
#define Kassiopeia_KSMathRK86_h_

#include "KSMathIntegrator.h"

#include <cmath>
#include <limits>

#define KSMATHRK86_STAGE 12

/*
* The basis for this ODE solver is given in;
*
* "Cheap Error Estimation For Runge-Kutta Methods"
*  Ch. Tsitouras and S.N. Papakostas
*  SIAM Journal on Scientific Computing, Volume 20,  Issue 6,  (November 1999)
*/

namespace Kassiopeia
{

template<class XSystemType> class KSMathRK86 : public KSMathIntegrator<XSystemType>
{
  public:
    KSMathRK86();
    ~KSMathRK86() override;

  public:
    typedef XSystemType SystemType;
    typedef KSMathDifferentiator<SystemType> DifferentiatorType;
    typedef typename SystemType::ValueType ValueType;
    typedef typename SystemType::DerivativeType DerivativeType;
    typedef typename SystemType::ErrorType ErrorType;

  public:
    void Integrate(double aTime, const DifferentiatorType& aTerm, const ValueType& anInitialValue, const double& aStep,
                   ValueType& aFinalValue, ErrorType& anError) const override;

    /*******************************************************************/
    void ClearState() override
    {
        fHaveCachedDerivative = false;
    };


    //returns true if information valid
    bool GetInitialDerivative(DerivativeType& derv) const override
    {
        if (fHaveCachedDerivative) {
            derv = fDerivatives[0];
            return true;
        }
        return false;
    };

    //returns true if information valid
    bool GetFinalDerivative(DerivativeType& derv) const override
    {
        if (fHaveCachedDerivative) {
            derv = fDerivatives[KSMATHRK86_STAGE];
            return true;
        }
        return false;
    };

    /******************************************************************/

  private:
    mutable bool fHaveCachedDerivative;
    mutable double fIntermediateTime[KSMATHRK86_STAGE + 1];
    mutable ValueType fValues[KSMATHRK86_STAGE + 1];
    mutable DerivativeType fDerivatives[KSMATHRK86_STAGE + 1];

    static const double fA[KSMATHRK86_STAGE][KSMATHRK86_STAGE];
    static const unsigned int fAColumnLimit[KSMATHRK86_STAGE];
    static const double fB6[KSMATHRK86_STAGE];
    static const double fB8[KSMATHRK86_STAGE];
    static const double fC[KSMATHRK86_STAGE];
};

template<class XSystemType> KSMathRK86<XSystemType>::KSMathRK86()
{
    for (unsigned int i = 0; i < KSMATHRK86_STAGE + 1; i++) {
        fIntermediateTime[i] = std::numeric_limits<double>::quiet_NaN();
        fValues[i] = std::numeric_limits<double>::quiet_NaN();
        fDerivatives[i] = std::numeric_limits<double>::quiet_NaN();
    }
    fHaveCachedDerivative = false;
}

template<class XSystemType> KSMathRK86<XSystemType>::~KSMathRK86() {}

template<class XSystemType>
void KSMathRK86<XSystemType>::Integrate(double aTime, const DifferentiatorType& aTerm, const ValueType& anInitialValue,
                                        const double& aStep, ValueType& aFinalValue, ErrorType& anError) const
{

    //do first stage (0) explicitly to deal with possibility of cached data

    //init value and time
    fValues[0] = anInitialValue;
    fIntermediateTime[0] = aTime;

    //init solution estimates
    ValueType y6 = fValues[0];
    ValueType y8 = fValues[0];

    //we check if we have cached the derivative from the last step
    if (fHaveCachedDerivative) {
        fDerivatives[0] = fDerivatives[KSMATHRK86_STAGE];
    }
    else {
        aTerm.Differentiate(fIntermediateTime[0], fValues[0], fDerivatives[0]);
    }

    //add contribution to 6th order estimate
    y6 = y6 + aStep * fB6[0] * fDerivatives[0];
    //add contribution to 8th order estimate
    y8 = y8 + aStep * fB8[0] * fDerivatives[0];

    //compute the value of each stage and
    //evaluation of the derivative at each stage
    for (unsigned int i = 1; i < KSMATHRK86_STAGE; i++) {
        //compute the time of this stage
        fIntermediateTime[i] = fIntermediateTime[0] + aStep * fC[i];

        //now compute the stage value
        fValues[i] = fValues[0];
        for (unsigned int j = 0; j < fAColumnLimit[i]; j++) {
            fValues[i] = fValues[i] + (aStep * fA[i][j]) * fDerivatives[j];
        }

        //now compute the derivative term for this stage
        aTerm.Differentiate(fIntermediateTime[i], fValues[i], fDerivatives[i]);

        //add contribution to 6th order estimate
        y6 = y6 + aStep * fB6[i] * fDerivatives[i];

        //add contribution to 8th order estimate
        y8 = y8 + aStep * fB8[i] * fDerivatives[i];
    }

    //we use the 8th order estimate for the solution (local extrapolation)
    aFinalValue = y8;

    //now estimate the truncation error on the step (for stepsize control)
    anError = (y6 - y8);

    //evaluate the derivative at final point and cache it for the next
    //step (this derivative is needed for the dense output interpolation)
    fIntermediateTime[KSMATHRK86_STAGE] = aTime + aStep;
    aTerm.Differentiate(fIntermediateTime[KSMATHRK86_STAGE], aFinalValue, fDerivatives[KSMATHRK86_STAGE]);
    fHaveCachedDerivative = true;


    return;
}

template<class XSystemType>
const double KSMathRK86<XSystemType>::fA[KSMATHRK86_STAGE][KSMATHRK86_STAGE] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},          //1st row
    {9.0 / 142.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  //2nd row
    {178422123.0 / 9178574137.0,
     685501333.0 / 8224473205.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},                                                                                       //3rd row
    {12257.0 / 317988.0, 0.0, 12257.0 / 105996.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},  //4th row
    {2584949729.0 / 6554704252.0,
     0.0,
     -9163901916.0 / 6184003973.0,
     26222057794.0 / 17776421907.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},  //5th row
    {4418011.0 / 96055225.0,
     0.0,
     0.0,
     2947922107.0 / 12687381736.0,
     3229973413.0 / 17234960414.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},  //6th row
    {2875139539.0 / 47877267651.0,
     0.0,
     0.0,
     2702377211.0 / 24084535832.0,
     -135707089.0 / 4042230341.0,
     299874140.0 / 17933325691.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},  //7th row
    {-7872176137.0 / 5003514694.0,
     0.0,
     0.0,
     -35136108789.0 / 26684798878.0,
     -114433184681.0 / 9760995895.0,
     299204996517.0 / 32851421233.0,
     254.0 / 39.0,
     0.0,
     0.0,
     0.0,
     0.0,
     0.0},  //8th row
    {-3559950777.0 / 7399971898.0,
     0.0,
     0.0,
     -29299291531.0 / 4405504148.0,
     -42434013379.0 / 9366905709.0,
     20642871700.0 / 5300635453.0,
     12951197050.0 / 1499985011.0,
     59527523.0 / 6331620793.0,
     0.0,
     0.0,
     0.0,
     0.0},  //9th row
    {-8196723582.0 / 10570795981.0,
     0.0,
     0.0,
     -46181454005.0 / 5775132776.0,
     -196277106011.0 / 29179424052.0,
     63575135343.0 / 11491868333.0,
     9348448139.0 / 857846776.0,
     195434294.0 / 9727139945.0,
     -617468037.0 / 15757346105.0,
     0.0,
     0.0,
     0.0},  //10th row
    {-6373809055.0 / 5357779452.0,
     0.0,
     0.0,
     -150772749657.0 / 21151088080.0,
     -58076657383.0 / 6089469394.0,
     9252721190.0 / 1221566797.0,
     132381309631.0 / 11748965576.0,
     704633904.0 / 13813696331.0,
     656417033.0 / 8185349658.0,
     -1669806516.0 / 10555289849.0,
     0.0,
     0.0},  //11th row
    {-2726346953.0 / 6954959789.0,
     0.0,
     0.0,
     24906446731.0 / 6359105161.0,
     -65277767625.0 / 23298960463.0,
     39128152317.0 / 16028215273.0,
     -40638357893.0 / 16804059016.0,
     -7437361171.0 / 21911114743.0,
     1040125706.0 / 5334949109.0,
     -1129865134.0 / 5812907645.0,
     6253441118.0 / 10543852725.0,
     0.0}  //12th row
};

//coefficients for the time-steps
template<class XSystemType>
const double KSMathRK86<XSystemType>::fC[KSMATHRK86_STAGE] = {0.0,
                                                              9.0 / 142.0,
                                                              24514.0 / 238491.0,
                                                              12257.0 / 79497.0,
                                                              50.0 / 129.0,
                                                              34.0 / 73.0,
                                                              23.0 / 148.0,
                                                              142.0 / 141.0,
                                                              14920944853.0 / 17030299364.0,
                                                              83.0 / 91.0,
                                                              143.0 / 149.0,
                                                              1.0};

template<class XSystemType>
const double KSMathRK86<XSystemType>::fB6[KSMATHRK86_STAGE] = {289283091.0 / 6008696510.0,
                                                               0.0,
                                                               0.0,
                                                               0.0,
                                                               0.0,
                                                               3034152487.0 / 7913336319.0,
                                                               7170564158.0 / 30263027435.0,
                                                               7206303747.0 / 16758195910.0,
                                                               -1059739258.0 / 8472387467.0,
                                                               16534129531.0 / 11550853505.0,
                                                               -3.0 / 2.0,
                                                               5118195927.0 / 53798651926.0};

template<class XSystemType>
const double KSMathRK86<XSystemType>::fB8[KSMATHRK86_STAGE] = {438853193.0 / 9881496838.0,
                                                               0.0,
                                                               0.0,
                                                               0.0,
                                                               0.0,
                                                               11093525429.0 / 31342013414.0,
                                                               481311443.0 / 1936695762.0,
                                                               -3375294558.0 / 10145424253.0,
                                                               9830993862.0 / 5116981057.0,
                                                               -138630849943.0 / 50747474617.0,
                                                               7152278206.0 / 5104393345.0,
                                                               5118195927.0 / 53798651926.0};

//list of the max column for each row in the fA matrix
//at which and beyond all entries are zero
template<class XSystemType>
const unsigned int KSMathRK86<XSystemType>::fAColumnLimit[KSMATHRK86_STAGE] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

}  // namespace Kassiopeia

#endif
