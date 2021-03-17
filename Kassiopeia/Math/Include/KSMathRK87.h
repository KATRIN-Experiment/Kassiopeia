#ifndef Kassiopeia_KSMathRK87_h_
#define Kassiopeia_KSMathRK87_h_

#include "KSMathIntegrator.h"

#include <limits>

#define KSMATHRK87_STAGE 13

/*
* The number of function evaluations per step is 13 for this method
*
* The basis for this ODE solver is given in;
*
* P.J. Prince and J.R. Dormand "High Order Embedded Runge-Kutta Formulae"
* Journal of Computational and Applied Mathematics. Vol. 7, pp. 67-75. Mar. 1981
*
*/

namespace Kassiopeia
{

template<class XSystemType> class KSMathRK87 : public KSMathIntegrator<XSystemType>
{
  public:
    KSMathRK87();
    ~KSMathRK87() override;

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
            derv = fDerivatives[KSMATHRK87_STAGE];
            return true;
        }
        return false;
    };

    //these functions are provided if the integrator implements
    //a method to interpolate the solution between initial and final step values
    //only valid for interpolating values on the last integration step
    bool HasDenseOutput() const override
    {
        return false;
    };
    void Interpolate(double /*aStepFraction*/, ValueType& /*aValue*/) const override;

    /******************************************************************/

  private:
    mutable bool fHaveCachedDerivative;
    mutable double fIntermediateTime[KSMATHRK87_STAGE + 1];
    mutable ValueType fValues[KSMATHRK87_STAGE + 1];
    mutable DerivativeType fDerivatives[KSMATHRK87_STAGE + 1];

    static const double fA[KSMATHRK87_STAGE][KSMATHRK87_STAGE];
    static const unsigned int fAColumnLimit[KSMATHRK87_STAGE];
    static const double fB7[KSMATHRK87_STAGE];
    static const double fB8[KSMATHRK87_STAGE];
    static const double fC[KSMATHRK87_STAGE];
};

template<class XSystemType> KSMathRK87<XSystemType>::KSMathRK87()
{
    for (unsigned int i = 0; i < KSMATHRK87_STAGE + 1; i++) {
        fIntermediateTime[i] = std::numeric_limits<double>::quiet_NaN();
        fValues[i] = std::numeric_limits<double>::quiet_NaN();
        fDerivatives[i] = std::numeric_limits<double>::quiet_NaN();
    }
    fHaveCachedDerivative = false;
}

template<class XSystemType> KSMathRK87<XSystemType>::~KSMathRK87() = default;

template<class XSystemType>
void KSMathRK87<XSystemType>::Integrate(double aTime, const DifferentiatorType& aTerm, const ValueType& anInitialValue,
                                        const double& aStep, ValueType& aFinalValue, ErrorType& anError) const
{
    //do first stage (0) explicitly to deal with possibility of cached data

    //init value and time
    fValues[0] = anInitialValue;
    fIntermediateTime[0] = aTime;

    //init solution estimates
    ValueType y7 = fValues[0];
    ValueType y8 = fValues[0];

    //we check if we have cached the derivative from the last step
    if (fHaveCachedDerivative) {
        fDerivatives[0] = fDerivatives[KSMATHRK87_STAGE];
    }
    else {
        aTerm.Differentiate(fIntermediateTime[0], fValues[0], fDerivatives[0]);
    }

    //add contribution to 7th order estimate
    y7 = y7 + aStep * fB7[0] * fDerivatives[0];
    //add contribution to 8th order estimate
    y8 = y8 + aStep * fB8[0] * fDerivatives[0];

    //compute the value of each stage and
    //evaluation of the derivative at each stage
    for (unsigned int i = 1; i < KSMATHRK87_STAGE; i++) {
        //compute the time of this stage
        fIntermediateTime[i] = fIntermediateTime[0] + aStep * fC[i];

        //now compute the stage value
        fValues[i] = fValues[0];
        for (unsigned int j = 0; j < fAColumnLimit[i]; j++) {
            fValues[i] = fValues[i] + (aStep * fA[i][j]) * fDerivatives[j];
        }

        //now compute the derivative term for this stage
        aTerm.Differentiate(fIntermediateTime[i], fValues[i], fDerivatives[i]);

        //add contribution to 7th order estimate
        y7 = y7 + aStep * fB7[i] * fDerivatives[i];

        //add contribution to 8th order estimate
        y8 = y8 + aStep * fB8[i] * fDerivatives[i];
    }

    //we use the 8th order estimate for the solution (local extrapolation)
    aFinalValue = y8;

    //now estimate the truncation error on the step (for stepsize control)
    anError = y7 - y8;

    //evaluate the derivative at final point and cache it for the next
    //step (this derivative is needed for the dense output interpolation)
    fIntermediateTime[KSMATHRK87_STAGE] = aTime + aStep;
    aTerm.Differentiate(fIntermediateTime[KSMATHRK87_STAGE], aFinalValue, fDerivatives[KSMATHRK87_STAGE]);
    fHaveCachedDerivative = true;

    return;
}

template<class XSystemType>
void KSMathRK87<XSystemType>::Interpolate(double /*aStepFraction*/, ValueType& /*aValue*/) const
{
    //TODO implement this interpolation function
}


template<class XSystemType>
const double KSMathRK87<XSystemType>::fA[KSMATHRK87_STAGE][KSMATHRK87_STAGE] = {
    {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},                       //1st row
    {1. / 18., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},                 //2nd row
    {1. / 48., 1. / 16., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},           //3rd row
    {1. / 32., 0., 3. / 32., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},           //4th row
    {5. / 16., 0., -75. / 64., 75. / 64., 0., 0., 0., 0., 0., 0., 0., 0., 0.},  //5th row
    {3. / 80., 0., 0., 3. / 16., 3. / 20., 0., 0., 0., 0., 0., 0., 0., 0.},     //6th row
    {29443841. / 614563906.,
     0.,
     0.,
     77736538. / 692538347.,
     -28693883. / 1125000000.,
     23124283. / 1800000000.,
     0.,
     0.,
     0.,
     0.,
     0.,
     0.,
     0.},  //7th row
    {16016141. / 946692911.,
     0.,
     0.,
     61564180. / 158732637.,
     22789713. / 633445777.,
     545815736. / 2771057229.,
     -180193667. / 1043307555.,
     0.,
     0.,
     0.,
     0.,
     0.,
     0.},  //8th row
    {39632708. / 573591083.,
     0.,
     0.,
     -433636366. / 683701615.,
     -421739975. / 2616292301.,
     100302831. / 723423059.,
     790204164. / 839813087.,
     800635310. / 3783071287.,
     0.,
     0.,
     0.,
     0.,
     0.},  //9th row
    {246121993. / 1340847787.,
     0.,
     0.,
     -37695042795. / 15268766246.,
     -309121744. / 1061227803.,
     -12992083. / 490766935.,
     6005943493. / 2108947869.,
     393006217. / 1396673457.,
     123872331. / 1001029789.,
     0.,
     0.,
     0.,
     0.},  //10th row
    {-1028468189. / 846180014.,
     0.,
     0.,
     8478235783. / 508512852.,
     1311729495. / 1432422823.,
     -10304129995. / 1701304382.,
     -48777925059. / 3047939560.,
     15336726248. / 1032824649.,
     -45442868181. / 3398467696.,
     3065993473. / 597172653.,
     0.,
     0.,
     0.},  //11th row
    {185892177. / 718116043.,
     0.,
     0.,
     -3185094517. / 667107341.,
     -477755414. / 1098053517.,
     -703635378. / 230739211.,
     5731566787. / 1027545527.,
     5232866602. / 850066563.,
     -4093664535. / 808688257.,
     3962137247. / 1805957418.,
     65686358. / 487910083.,
     0.,
     0.},  //12th row
    {403863854. / 491063109.,
     0.,
     0.,
     -5068492393. / 434740067.,
     -411421997. / 543043805.,
     652783627. / 914296604.,
     11173962825. / 925320556.,
     -13158990841. / 6184727034.,
     3936647629. / 1978049680.,
     -160528059. / 685178525.,
     248638103. / 1413531060.,
     0.,
     0.}  //13th row
};

template<class XSystemType>
const double KSMathRK87<XSystemType>::fB7[KSMATHRK87_STAGE] = {13451932. / 455176623.,
                                                               0.,
                                                               0.,
                                                               0.,
                                                               0.,
                                                               -808719846. / 976000145.,
                                                               1757004468. / 5645159321.,
                                                               656045339. / 265891186.,
                                                               -3867574721. / 1518517206.,
                                                               465885868. / 322736535.,
                                                               53011238. / 667516719.,
                                                               2. / 45.,
                                                               0.};

template<class XSystemType>
const double KSMathRK87<XSystemType>::fB8[KSMATHRK87_STAGE] = {14005451. / 335480064.,
                                                               0.,
                                                               0.,
                                                               0.,
                                                               0.,
                                                               -59238493. / 1068277825.,
                                                               181606767. / 758867731.,
                                                               561292985. / 797845732.,
                                                               -1041891430. / 1371343529.,
                                                               760417239. / 1151165299.,
                                                               118820643. / 751138087.,
                                                               -528747749. / 2220607170.,
                                                               1. / 4.};

template<class XSystemType>
const double KSMathRK87<XSystemType>::fC[KSMATHRK87_STAGE] = {0.0,
                                                              1.0 / 18.0,
                                                              1.0 / 12.,
                                                              1.0 / 8.0,
                                                              5.0 / 16.0,
                                                              3.0 / 8.0,
                                                              59.0 / 400.0,
                                                              93.0 / 200.,
                                                              5490023248. / 9719169821.,
                                                              13. / 20.,
                                                              1201146811. / 1299019798.,
                                                              1.,
                                                              1.};

//list of the max column for each row in the fA matrix
//at which and beyond all entries are zero
template<class XSystemType>
const unsigned int KSMathRK87<XSystemType>::fAColumnLimit[KSMATHRK87_STAGE] =
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11};

}  // namespace Kassiopeia

#endif
