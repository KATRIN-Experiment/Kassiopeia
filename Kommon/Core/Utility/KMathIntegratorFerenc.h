/**
 * @file KMathIntegratorFerenc.h
 *
 * @date 02.01.2014
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 * @author Wolfgang Käfer
 *
 * @brief template class for numerical integration, based on a C function from Ferenc
 * @details
 * <b> usage <b>
 * The Integrator can integrate either a free "C-style" function with signatur double func(double x)
 * or, more useful, a "functionoid" class which implements the double operator()(double x).
 * the function or an instance of the class can be passed as template argument.
 *
 * Additional parameters can be passed to the class beforehand, either in the constructor or via setters...
 *
 * \code
 * double LinearFunc(double x){
 *    return x;
 * }
 *
 * class QuadraticFunc{
 *   public:
 *     QuadraticFunc(double p0 = 1., double p1 = 2., double p2 =2.):
 *           fp0(p0),
 *           fp1(p1),
 *           fp2(p2)
 *     {}
 *     ~QuadraticFunc(){};
 *
 *     double operator()(double x){
 *              return fp0 + fp1 * x + fp2* x*x;
 *     }
 *
 *     protected:
 *         double fp0, fp1, fp2;
 *
 * };
 *
 * int main (){
 *    KMathIntegratorFerenc Integrator;
 *    Integrator.SetRange(0., 1.);
 *    Integrator.SetNSteps(25);
 *    double resultlin = Integrator.Integrate(LinearFunc);
 *    QuadraticFunc defaultquadraticclass;
 *    double resultquad =Integrator.Integrate(defaultquadraticthing);
 *  }
 * \endcode
 *
 */

#ifndef KMATHINTEGRATORFERENC_H_
#define KMATHINTEGRATORFERENC_H_


#include <assert.h>
#include <algorithm>
#include <cmath>

namespace katrin {

class KMathIntegratorFerenc {
public:
    static double GetWeight(uint32_t iStep, uint32_t NSteps);

public:
    KMathIntegratorFerenc(double xMin = 0.0, double xMax = 1.0, uint32_t nSteps = 25) :
        fXMin(xMin), fXMax(xMax), fNSteps(nSteps)
    {
        SetNSteps(nSteps);
    }

    template <typename TFunc>
    double Integrate( TFunc&  func) const;

    template <typename TFunc>
    double Integrate( const TFunc&  func) const;

    void SetRange(double xmin, double xmax) {
        fXMin = xmin;
        fXMax = xmax;
    }
    void SetNSteps(uint32_t nsteps) {
        /*if (nsteps <= 1) fNSteps = 1;
        else */if (nsteps <= 6) fNSteps = 6;
        else if (nsteps <= 12) fNSteps = 12;
        else fNSteps = nsteps;
    }
    uint32_t GetNSteps() const {
        return fNSteps;
    }
    void SetStepSize(double stepSize) {
        SetNSteps( (uint32_t) fabs((fXMax-fXMin) / stepSize) + 1 );
    }
    double GetStepSize() const {
        return fabs( (fXMax-fXMin) / (double) fNSteps );
    }

protected:
    double fXMin;
    double fXMax;
    uint32_t fNSteps;
};


inline double KMathIntegratorFerenc::GetWeight(uint32_t iStep, uint32_t NSteps)
{
    // NSteps == 1
    static const double sWeightSingle = 0.5;

    // NSteps == 6
    static const double sWeightsWeddle[7] = {
        3.0/10.0, 3.0/2.0, 3.0/10.0, 9.0/5.0, 3.0/10.0, 3.0/2.0, 3.0/10.0
    };

    // NSteps >= 12
    static const double sWeights5[6] = {
        0.3187500000000000e+00, 0.1376388888888889e+01,
        0.6555555555555556e+00, 0.1212500000000000e+01,
        0.9256944444444445e+00, 0.1011111111111111e+01
    };

    // NSteps >= 20
    static const double sWeights9[10] = {
        0.2803440531305107e0, 0.1648702325837748e1,
        -0.2027449845679092e0, 0.2797927414021179e1,
        -0.9761199294532843e0, 0.2556499393738999e1,
        0.1451083002645404e0, 0.1311227127425048e1,
        0.9324249063051143e0, 0.1006631393298060e1
    };

    assert( (NSteps == 1 || NSteps == 6 || NSteps >= 12) && iStep <= NSteps );

    if (NSteps == 1) {
        return sWeightSingle;
    }
    if (NSteps == 6) {
        return sWeightsWeddle[iStep];
    }
    else if (NSteps < 20) {
        if (iStep <= 5)
            return sWeights5[iStep];
        else if (iStep >= NSteps-5)
            return sWeights5[NSteps-iStep];
        else
            return 1.0;
    }
    else {
        if (iStep <= 9)
            return sWeights9[iStep];
        else if (iStep >= NSteps-9)
            return sWeights9[NSteps-iStep];
        else
            return 1.0;
    }
}


template <typename TFunc>
inline double KMathIntegratorFerenc::Integrate( TFunc& func) const
{
    if (fXMax == fXMin)
        return 0.0;

    double integral = 0.0;
    const double stepsize = (fXMax - fXMin) / (double) fNSteps;

    for (uint32_t iStep = 0; iStep <= fNSteps; ++iStep) {
        integral += GetWeight(iStep, fNSteps) * func( fXMin + stepsize * (double) iStep );
    }

    return integral * stepsize;
}

template <typename TFunc>
inline double KMathIntegratorFerenc::Integrate( const TFunc&  func) const
{
    return this->Integrate(const_cast<TFunc&>(func));
}

}

#endif /* KMATHINTEGRATORFERENC_H_ */
