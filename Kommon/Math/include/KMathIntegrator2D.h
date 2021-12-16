/**
 * @file KMathIntegrator2D.h
 *
 * @date 26.09.2014
 * @author Marco Haag <marco.haag@kit.edu>
 */
#ifndef KMATHINTEGRATOR2D_H_
#define KMATHINTEGRATOR2D_H_

#include "KMathIntegrator.h"

namespace katrin
{

template<class XFloatT, class XSamplingPolicy = policies::PlainSumming> class KMathIntegrator2D
{
  public:
    KMathIntegrator2D(XFloatT precision = 1E-4, KEMathIntegrationMethod method = KEMathIntegrationMethod::Simpson);
    virtual ~KMathIntegrator2D() = default;

    /**
     * Set the integration limits.
     * @param xStart
     * @param xEnd
     */
    void SetXRange(XFloatT xStart, XFloatT xEnd)
    {
        xIntegrator.SetRange(xStart, xEnd);
    }
    std::pair<XFloatT, XFloatT> GetXRange() const
    {
        return xIntegrator.GetRange();
    }

    /**
     * Set the integration limits.
     * @param yStart
     * @param yEnd
     */
    void SetYRange(XFloatT yStart, XFloatT yEnd)
    {
        yIntegrator.SetRange(yStart, yEnd);
    }
    std::pair<XFloatT, XFloatT> GetYRange() const
    {
        return yIntegrator.GetRange();
    }

    /**
     * Set the integration limits.
     * @param yStart
     * @param yEnd
     */
    void SetRange(XFloatT xStart, XFloatT xEnd, XFloatT yStart, XFloatT yEnd)
    {
        xIntegrator.SetRange(xStart, xEnd);
        yIntegrator.SetRange(yStart, yEnd);
    }

    /**
     * Set the desired precision.
     * @param precision
     */
    void SetPrecision(XFloatT precision)
    {
        xIntegrator.SetPrecision(precision);
        yIntegrator.SetPrecision(precision);
    }
    XFloatT GetPrecision() const
    {
        return xIntegrator.GetPrecision();
    }

    uint32_t SetMinSteps(int32_t min)
    {
        xIntegrator.SetMinSteps(min);
        return yIntegrator.SetMinSteps(min);
    }
    uint32_t GetMinSteps() const
    {
        return xIntegrator.GetMinSteps();
    }
    uint32_t SetMaxSteps(int32_t max)
    {
        xIntegrator.SetMaxSteps(max);
        return yIntegrator.SetMaxSteps(max);
    }
    uint32_t GetMaxSteps() const
    {
        return xIntegrator.GetMaxSteps();
    }

    /**
     * Fix the minimum and maximum number of steps to the same value.
     * @param minAndMax
     * @return
     */
    uint32_t SetSteps(int32_t minAndMax)
    {
        SetMinSteps(minAndMax);
        return SetMaxSteps(minAndMax);
    }

    /**
     * Configure the integration algorithm.
     * @param method
     */
    void SetMethod(KEMathIntegrationMethod method)
    {
        xIntegrator.SetMethod(method);
        yIntegrator.SetMethod(method);
    }
    KEMathIntegrationMethod GetMethod() const
    {
        return xIntegrator.GetMethod();
    }

    void ThrowExceptions(bool throwExc)
    {
        xIntegrator.ThrowExceptions(throwExc);
        yIntegrator.ThrowExceptions(throwExc);
    }
    void FallbackOnLowPrecision(bool fallback)
    {
        xIntegrator.FallbackOnLowPrecision(fallback);
        yIntegrator.FallbackOnLowPrecision(fallback);
    }

    uint32_t NumberOfIterations() const
    {
        return fIterations;
    }
    uint64_t NumberOfSteps() const
    {
        return fSteps;
    }

    /**
     * Perform the integration.
      * @return
     */
    template<class XIntegrandType2D> XFloatT Integrate(XIntegrandType2D& func);

    /**
     * Perform the integration.
     * @param xStart Lower integration limit.
     * @param xEnd Upper integration limit.
     * @param yStart Lower integration limit.
     * @param yEnd Upper integration limit.
     * @return
     */
    template<class XIntegrandType2D>
    XFloatT Integrate(XIntegrandType2D& func, XFloatT xStart, XFloatT xEnd, XFloatT yStart, XFloatT yEnd);

  protected:
    KMathIntegrator<XFloatT, XSamplingPolicy> xIntegrator;
    KMathIntegrator<XFloatT, XSamplingPolicy> yIntegrator;
    uint32_t fIterations;
    uint64_t fSteps;

    template<class XIntegrandType2D> struct InnerFunctor
    {
        InnerFunctor(XIntegrandType2D& func) : fX(0.0), fIntegrand(func) {}
        XFloatT operator()(XFloatT y)
        {
            return fIntegrand(fX, y);
        }
        XFloatT fX;
        XIntegrandType2D& fIntegrand;
    };

    template<class XIntegrandType2D> struct OuterFunctor
    {
        OuterFunctor(XIntegrandType2D& func, KMathIntegrator<XFloatT, XSamplingPolicy>& integrator,
                     uint32_t& iterationCounter, uint64_t& stepCounter) :
            fInnerFunctor(func),
            fIntegrator(integrator),
            fIterations(iterationCounter),
            fSteps(stepCounter)
        {}
        XFloatT operator()(XFloatT x)
        {
            fInnerFunctor.fX = x;
            const XFloatT result = fIntegrator.Integrate(fInnerFunctor);
            fIterations += fIntegrator.NumberOfIterations();
            fSteps += fIntegrator.NumberOfSteps();
            return result;
        }
        InnerFunctor<XIntegrandType2D> fInnerFunctor;
        KMathIntegrator<XFloatT, XSamplingPolicy>& fIntegrator;
        uint32_t& fIterations;
        uint64_t& fSteps;
    };
};

template<class XFloatT, class XSamplingPolicy>
inline KMathIntegrator2D<XFloatT, XSamplingPolicy>::KMathIntegrator2D(XFloatT precision,
                                                                      KEMathIntegrationMethod method) :
    xIntegrator(precision, method),
    yIntegrator(precision, method),
    fIterations(0),
    fSteps(0)
{}

template<class XFloatT, class XSamplingPolicy>
template<class XIntegrandType2D>
inline XFloatT KMathIntegrator2D<XFloatT, XSamplingPolicy>::Integrate(XIntegrandType2D& integrand)
{
    fIterations = fSteps = 0;
    OuterFunctor<XIntegrandType2D> outerFunctor(integrand, yIntegrator, fIterations, fSteps);
    return xIntegrator.Integrate(outerFunctor);
}

template<class XFloatT, class XSamplingPolicy>
template<class XIntegrandType2D>
inline XFloatT KMathIntegrator2D<XFloatT, XSamplingPolicy>::Integrate(XIntegrandType2D& func, XFloatT xStart,
                                                                      XFloatT xEnd, XFloatT yStart, XFloatT yEnd)
{
    SetXRange(xStart, xEnd);
    SetYRange(yStart, yEnd);
    return Integrate(func);
}

} /* namespace katrin */

#endif /* KMATHINTEGRATOR2D_H_ */
