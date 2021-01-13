#ifndef Kassiopeia_KSMathRK65_h_
#define Kassiopeia_KSMathRK65_h_

#include "KSMathIntegrator.h"

#include <cmath>
#include <limits>

#define KSMATHRK65_STAGE 8

/*
* The number of function evaluations per step is 8 for this method.
*
* The basis for this ODE solver is given in;
*
* P.J. Prince and J.R. Dormand "High Order Embedded Runge-Kutta Formulae"
* Journal of Computational and Applied Mathematics. Vol. 7, pp. 67-75. Mar. 1981
*
*/

namespace Kassiopeia
{

template<class XSystemType> class KSMathRK65 : public KSMathIntegrator<XSystemType>
{
  public:
    KSMathRK65();
    ~KSMathRK65() override;

  public:
    typedef XSystemType SystemType;
    using DifferentiatorType = KSMathDifferentiator<SystemType>;
    using ValueType = typename SystemType::ValueType;
    using DerivativeType = typename SystemType::DerivativeType;
    using ErrorType = typename SystemType::ErrorType;

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
            derv = fDerivatives[KSMATHRK65_STAGE];
            return true;
        }
        return false;
    };

    /******************************************************************/


  private:
    mutable bool fHaveCachedDerivative;
    mutable double fIntermediateTime[KSMATHRK65_STAGE + 1];
    mutable ValueType fValues[KSMATHRK65_STAGE + 1];
    mutable DerivativeType fDerivatives[KSMATHRK65_STAGE + 1];

    //parameters defining the Butcher Tableau
    static const double fA[KSMATHRK65_STAGE][KSMATHRK65_STAGE];
    static const unsigned int fAColumnLimit[KSMATHRK65_STAGE];
    static const double fB5[KSMATHRK65_STAGE];
    static const double fB6[KSMATHRK65_STAGE];
    static const double fC[KSMATHRK65_STAGE];
};

template<class XSystemType> KSMathRK65<XSystemType>::KSMathRK65()
{
    for (unsigned int i = 0; i < KSMATHRK65_STAGE + 1; i++) {
        fIntermediateTime[i] = std::numeric_limits<double>::quiet_NaN();
        fValues[i] = std::numeric_limits<double>::quiet_NaN();
        fDerivatives[i] = std::numeric_limits<double>::quiet_NaN();
    }
    fHaveCachedDerivative = false;
}

template<class XSystemType> KSMathRK65<XSystemType>::~KSMathRK65() = default;

template<class XSystemType>
void KSMathRK65<XSystemType>::Integrate(double aTime, const DifferentiatorType& aTerm, const ValueType& anInitialValue,
                                        const double& aStep, ValueType& aFinalValue, ErrorType& anError) const
{
    //do first stage (0) explicitly to deal with possibility of cached data

    //init value and time
    fValues[0] = anInitialValue;
    fIntermediateTime[0] = aTime;

    //init solution estimates
    ValueType y5 = fValues[0];
    ValueType y6 = fValues[0];

    //we check if we have cached the derivative from the last step
    if (fHaveCachedDerivative) {
        fDerivatives[0] = fDerivatives[KSMATHRK65_STAGE];
    }
    else {
        aTerm.Differentiate(fIntermediateTime[0], fValues[0], fDerivatives[0]);
    }

    //add contribution to 5th order estimate
    y5 = y5 + aStep * fB5[0] * fDerivatives[0];
    //add contribution to 6th order estimate
    y6 = y6 + aStep * fB6[0] * fDerivatives[0];

    //compute the value of each stage and
    //evaluation of the derivative at each stage
    for (unsigned int i = 1; i < KSMATHRK65_STAGE; i++) {
        //compute the time of this stage
        fIntermediateTime[i] = fIntermediateTime[0] + aStep * fC[i];

        //now compute the stage value
        fValues[i] = fValues[0];
        for (unsigned int j = 0; j < fAColumnLimit[i]; j++) {
            fValues[i] = fValues[i] + (aStep * fA[i][j]) * fDerivatives[j];
        }

        //now compute the derivative term for this stage
        aTerm.Differentiate(fIntermediateTime[i], fValues[i], fDerivatives[i]);

        //add contribution to 5th order estimate
        y5 = y5 + aStep * fB5[i] * fDerivatives[i];

        //add contribution to 6th order estimate
        y6 = y6 + aStep * fB6[i] * fDerivatives[i];
    }

    //we use the 6th order estimate for the solution (local extrapolation)
    aFinalValue = y6;

    //now estimate the truncation error on the step (for stepsize control)
    anError = y5 - y6;

    //evaluate the derivative at final point and cache it for the next
    //step (this derivative is needed for the dense output interpolation)
    fIntermediateTime[KSMATHRK65_STAGE] = aTime + aStep;
    aTerm.Differentiate(fIntermediateTime[KSMATHRK65_STAGE], aFinalValue, fDerivatives[KSMATHRK65_STAGE]);
    fHaveCachedDerivative = true;

    return;
}

template<class XSystemType>
const double KSMathRK65<XSystemType>::fA[KSMATHRK65_STAGE][KSMATHRK65_STAGE] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1. / 10., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {-2. / 81., 20. / 81., 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {615. / 1372., -270. / 343., 1053. / 1372., 0., 0., 0., 0., 0.},
    {3243. / 5500., -54. / 55., 50949. / 71500., 4998. / 17875., 0., 0., 0., 0.},
    {-26492. / 37125., 72. / 55., 2808. / 23375., -24206. / 37125., 338. / 459., 0., 0., 0.},
    {5561. / 2376., -35. / 11., -24117. / 31603., 899983. / 200772., -5225. / 1836., 3925. / 4056., 0., 0.},
    {465467. / 266112.,
     -2945. / 1232.,
     -5610201. / 14158144.,
     10513573. / 3212352.,
     -424325. / 205632.,
     376225. / 454272.,
     0.,
     0.}};

//coefficients for the time-steps
template<class XSystemType>
const double KSMathRK65<XSystemType>::fC[KSMATHRK65_STAGE] = {0., 1. / 10., 2. / 9., 3. / 7., 3. / 5., 4. / 5., 1., 1.};

template<class XSystemType>
const double KSMathRK65<XSystemType>::fB5[KSMATHRK65_STAGE] =
    {821. / 10800., 0., 19683. / 71825., 175273. / 912600., 395. / 3672., 785. / 2704., 3. / 50., 0.};

template<class XSystemType>
const double KSMathRK65<XSystemType>::fB6[KSMATHRK65_STAGE] =
    {61. / 864., 0., 98415. / 321776., 16807. / 146016., 1375. / 7344., 1375. / 5408., -37. / 1120., 1. / 10.};

//list of the max column for each row in the fA matrix
//at which and beyond all entries are zero
template<class XSystemType>
const unsigned int KSMathRK65<XSystemType>::fAColumnLimit[KSMATHRK65_STAGE] = {0, 1, 2, 3, 4, 5, 6, 6};

}  // namespace Kassiopeia

#endif
