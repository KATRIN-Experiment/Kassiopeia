#ifndef Kassiopeia_KSMathRK8_h_
#define Kassiopeia_KSMathRK8_h_

#include "KSMathIntegrator.h"

/* This integrator is time independent */

namespace Kassiopeia
{

template<class XSystemType> class KSMathRK8 : public KSMathIntegrator<XSystemType>
{
  public:
    typedef XSystemType SystemType;
    using DifferentiatorType = KSMathDifferentiator<SystemType>;
    using ValueType = typename SystemType::ValueType;
    using DerivativeType = typename SystemType::DerivativeType;
    using ErrorType = typename SystemType::ErrorType;

  public:
    KSMathRK8();
    ~KSMathRK8() override;

  public:
    void Integrate(double /*aTime*/, const DifferentiatorType& aTerm, const ValueType& anInitialValue,
                   const double& aStep, ValueType& aFinalValue, ErrorType& anError) const override;

  private:
    enum
    {
        eSubsteps = 13,
        eConditions = eSubsteps - 1
    };

    mutable ValueType fValues[eConditions];
    mutable DerivativeType fDerivatives[eSubsteps];

    static const double fA[eSubsteps][eSubsteps];
};

template<class XSystemType> KSMathRK8<XSystemType>::KSMathRK8() = default;

template<class XSystemType> KSMathRK8<XSystemType>::~KSMathRK8() = default;

template<class XSystemType>
void KSMathRK8<XSystemType>::Integrate(double /*aTime*/, const DifferentiatorType& aTerm,
                                       const ValueType& anInitialValue, const double& aStep, ValueType& aFinalValue,
                                       ErrorType& /*anError*/) const
{
    double dummyTime = 0;

    aTerm.Differentiate(dummyTime, anInitialValue, fDerivatives[0]);

    fValues[0] = anInitialValue + aStep * (fA[0][0] * fDerivatives[0]);

    aTerm.Differentiate(dummyTime, fValues[0], fDerivatives[1]);

    fValues[1] = anInitialValue + aStep * (fA[1][0] * fDerivatives[0] + fA[1][1] * fDerivatives[1]);

    aTerm.Differentiate(dummyTime, fValues[1], fDerivatives[2]);

    fValues[2] =
        anInitialValue + aStep * (fA[2][0] * fDerivatives[0] + fA[2][1] * fDerivatives[1] + fA[2][2] * fDerivatives[2]);

    aTerm.Differentiate(dummyTime, fValues[2], fDerivatives[3]);

    fValues[3] = anInitialValue + aStep * (fA[3][0] * fDerivatives[0] + fA[3][1] * fDerivatives[1] +
                                           fA[3][2] * fDerivatives[2] + fA[3][3] * fDerivatives[3]);

    aTerm.Differentiate(dummyTime, fValues[3], fDerivatives[4]);

    fValues[4] =
        anInitialValue + aStep * (fA[4][0] * fDerivatives[0] + fA[4][1] * fDerivatives[1] + fA[4][2] * fDerivatives[2] +
                                  fA[4][3] * fDerivatives[3] + fA[4][4] * fDerivatives[4]);

    aTerm.Differentiate(dummyTime, fValues[4], fDerivatives[5]);

    fValues[5] =
        anInitialValue + aStep * (fA[5][0] * fDerivatives[0] + fA[5][1] * fDerivatives[1] + fA[5][2] * fDerivatives[2] +
                                  fA[5][3] * fDerivatives[3] + fA[5][4] * fDerivatives[4] + fA[5][5] * fDerivatives[5]);

    aTerm.Differentiate(dummyTime, fValues[5], fDerivatives[6]);

    fValues[6] =
        anInitialValue + aStep * (fA[6][0] * fDerivatives[0] + fA[6][1] * fDerivatives[1] + fA[6][2] * fDerivatives[2] +
                                  fA[6][3] * fDerivatives[3] + fA[6][4] * fDerivatives[4] + fA[6][5] * fDerivatives[5] +
                                  fA[6][6] * fDerivatives[6]);

    aTerm.Differentiate(dummyTime, fValues[6], fDerivatives[7]);

    fValues[7] =
        anInitialValue + aStep * (fA[7][0] * fDerivatives[0] + fA[7][1] * fDerivatives[1] + fA[7][2] * fDerivatives[2] +
                                  fA[7][3] * fDerivatives[3] + fA[7][4] * fDerivatives[4] + fA[7][5] * fDerivatives[5] +
                                  fA[7][6] * fDerivatives[6] + fA[7][7] * fDerivatives[7]);

    aTerm.Differentiate(dummyTime, fValues[7], fDerivatives[8]);

    fValues[8] =
        anInitialValue + aStep * (fA[8][0] * fDerivatives[0] + fA[8][1] * fDerivatives[1] + fA[8][2] * fDerivatives[2] +
                                  fA[8][3] * fDerivatives[3] + fA[8][4] * fDerivatives[4] + fA[8][5] * fDerivatives[5] +
                                  fA[8][6] * fDerivatives[6] + fA[8][7] * fDerivatives[7] + fA[8][8] * fDerivatives[8]);

    aTerm.Differentiate(dummyTime, fValues[8], fDerivatives[9]);

    fValues[9] =
        anInitialValue + aStep * (fA[9][0] * fDerivatives[0] + fA[9][1] * fDerivatives[1] + fA[9][2] * fDerivatives[2] +
                                  fA[9][3] * fDerivatives[3] + fA[9][4] * fDerivatives[4] + fA[9][5] * fDerivatives[5] +
                                  fA[9][6] * fDerivatives[6] + fA[9][7] * fDerivatives[7] + fA[9][8] * fDerivatives[8] +
                                  fA[9][9] * fDerivatives[9]);

    aTerm.Differentiate(dummyTime, fValues[9], fDerivatives[10]);

    fValues[10] = anInitialValue +
                  aStep * (fA[10][0] * fDerivatives[0] + fA[10][1] * fDerivatives[1] + fA[10][2] * fDerivatives[2] +
                           fA[10][3] * fDerivatives[3] + fA[10][4] * fDerivatives[4] + fA[10][5] * fDerivatives[5] +
                           fA[10][6] * fDerivatives[6] + fA[10][7] * fDerivatives[7] + fA[10][8] * fDerivatives[8] +
                           fA[10][9] * fDerivatives[9] + fA[10][10] * fDerivatives[10]);

    aTerm.Differentiate(dummyTime, fValues[10], fDerivatives[11]);

    fValues[11] = anInitialValue +
                  aStep * (fA[11][0] * fDerivatives[0] + fA[11][1] * fDerivatives[1] + fA[11][2] * fDerivatives[2] +
                           fA[11][3] * fDerivatives[3] + fA[11][4] * fDerivatives[4] + fA[11][5] * fDerivatives[5] +
                           fA[11][6] * fDerivatives[6] + fA[11][7] * fDerivatives[7] + fA[11][8] * fDerivatives[8] +
                           fA[11][9] * fDerivatives[9] + fA[11][10] * fDerivatives[10] + fA[11][11] * fDerivatives[11]);

    aTerm.Differentiate(dummyTime, fValues[11], fDerivatives[12]);

    aFinalValue = anInitialValue +
                  aStep * (fA[12][0] * fDerivatives[0] + fA[12][1] * fDerivatives[1] + fA[12][2] * fDerivatives[2] +
                           fA[12][3] * fDerivatives[3] + fA[12][4] * fDerivatives[4] + fA[12][5] * fDerivatives[5] +
                           fA[12][6] * fDerivatives[6] + fA[12][7] * fDerivatives[7] + fA[12][8] * fDerivatives[8] +
                           fA[12][9] * fDerivatives[9] + fA[12][10] * fDerivatives[10] + fA[12][11] * fDerivatives[11] +
                           fA[12][12] * fDerivatives[12]);

    return;
}

template<class XSystemType>
const double KSMathRK8<XSystemType>::fA[eSubsteps][eSubsteps] = {
    {1. / 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},                                               //1st row
    {5. / 72., 1. / 72., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},                                        //2nd row
    {1. / 32., 0., 3. / 32., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.},                                        //3rd row
    {106. / 125., 0., -408. / 125., 352. / 125., 0., 0., 0., 0., 0., 0., 0., 0., 0.},                        //4th row
    {1. / 48., 0., 0., 8. / 33., 125. / 528., 0., 0., 0., 0., 0., 0., 0., 0.},                               //5th row
    {-1263. / 2401., 0., 0., 39936. / 26411., -64125. / 26411., 5520. / 2401., 0., 0., 0., 0., 0., 0., 0.},  //6th row
    {37. / 392., 0., 0., 0., 1625. / 9408., -2. / 15., 61. / 6720., 0., 0., 0., 0., 0., 0.},                 //7th row
    {17176. / 25515.,
     0.,
     0.,
     -47104. / 25515.,
     1325. / 504.,
     -41792. / 25515.,
     20237. / 145800.,
     4312. / 6075.,
     0.,
     0.,
     0.,
     0.,
     0.},  //8th row
    {-23834. / 180075.,
     0.,
     0.,
     -77824. / 1980825.,
     -636635. / 633864.,
     254048. / 300125.,
     -183. / 7000.,
     8. / 11.,
     -324. / 3773.,
     0.,
     0.,
     0.,
     0.},  //9th row
    {12733. / 7600.,
     0.,
     0.,
     -20032. / 5225.,
     456485. / 80256.,
     -42599. / 7125.,
     339227. / 912000.,
     -1029. / 4180.,
     1701. / 1408.,
     5145. / 2432.,
     0.,
     0.,
     0.},  //10th row
    {-27061. / 204120.,
     0.,
     0.,
     40448. / 280665.,
     -1353775. / 1197504.,
     17662. / 25515.,
     -71687. / 1166400.,
     98. / 225.,
     1. / 16.,
     3773. / 11664.,
     0.,
     0.,
     0.},  //11th row
    {11203. / 8680.,
     0.,
     0.,
     -38144. / 11935.,
     2354425. / 458304.,
     -84046. / 16275.,
     673309. / 1636800.,
     4704. / 8525.,
     9477. / 10912.,
     -1029. / 992.,
     0.,
     729. / 341.,
     0.},  //12th row
    {31. / 720.,
     0.,
     0.,
     0.,
     0.,
     16. / 75.,
     16807. / 79200.,
     16807. / 79200.,
     243. / 1760.,
     0.,
     0.,
     243. / 1760.,
     31. / 720.}  //13th row
};

}  // namespace Kassiopeia

#endif
