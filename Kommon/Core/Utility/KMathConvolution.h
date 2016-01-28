/**
 * @file KMathConvolution.h
 *
 * @date 27.12.2013
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KMATHCONVOLUTION_H_
#define KMATHCONVOLUTION_H_

#include "KMathIntegrator.h"
#include "KConst.h"

namespace katrin
{

template<class XKernelType>
class KMathConvolution
{
public:
    KMathConvolution(const XKernelType& kernel = XKernelType());
    virtual ~KMathConvolution() { }

    XKernelType& GetKernel() { return fKernel; }
    KMathIntegrator& GetIntegrator() { return fIntegrator; }

    template<class XFunctionType>
    double Convolute(XFunctionType& function, double x);

protected:
    KMathIntegrator fIntegrator;
    XKernelType fKernel;

    template<class XFunctionType>
    struct Integrand
    {
        Integrand(XFunctionType& function, XKernelType& kernel, double x) :
            fFunction(function), fKernel(kernel), fX(x) { }
        XFunctionType& fFunction;
        XKernelType& fKernel;
        double fX;
        double operator()(double tau) {
            const double f = fFunction(fX-tau);
            if (f == 0.0)
                return 0.0;
            const double k = fKernel(tau);
            return f * k;
        }
    };
};

template<class XKernelType>
inline KMathConvolution<XKernelType>::KMathConvolution(const XKernelType& kernel) :
    fIntegrator(),
    fKernel( kernel )
{
    fIntegrator.SetPrecision(1E-5);
    fIntegrator.SetMinSteps(16);
    fIntegrator.SetMethod( KEMathIntegrationMethod::Romberg );
}

template<class XKernelType>
template<class XFunctionType>
inline double KMathConvolution<XKernelType>::Convolute(XFunctionType& function, double x)
{
    Integrand<XFunctionType> integrand(function, fKernel, x);
    const std::pair<double, double> kernelRange = fKernel.GetRange();
    fIntegrator.SetRange( kernelRange.first, kernelRange.second );
    const double result = fIntegrator.Integrate( integrand ); // / (kernelRange.second - kernelRange.first);
    return result;
}


/*** KERNELS ***/

class KMathGaussKernel
{
public:
    KMathGaussKernel(double mean = 0.0, double sigma = 1.0) :
        fMean(mean), fSigma(sigma) { }

    void SetMean(double value) { fMean = value; }
    double GetMean() const { return fMean; }

    void SetSigma(double value) { fSigma = value; }
    double GetSigma() const { return fSigma; }

    std::pair<double, double> GetRange() const {
        return std::make_pair(fMean - 5.0 * fSigma, fMean + 5.0 * fSigma);
    }

    double operator()(double x) const {
        double exponent = x - fMean;
        exponent *= -exponent;
        exponent /= 2.0 * fSigma * fSigma;
        double result = exp(exponent);
        result /= fSigma * sqrt(2.0 * KConst::Pi()) * 0.9999994266968563;   // normalize by 5 sigma p value
        return result;
    }

private:
    double fMean;
    double fSigma;
};


} /* namespace katrin */

#endif /* KMATHCONVOLUTION_H_ */
