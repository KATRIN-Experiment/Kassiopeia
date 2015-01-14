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

class KMathIntegrator2D
{
public:
    KMathIntegrator2D(double precision = 1E-4, KMathIntegrator::EIntegrationMethod method = KMathIntegrator::eSimpson);
    virtual ~KMathIntegrator2D() { };

    /**
     * Set the integration limits.
     * @param xStart
     * @param xEnd
     */
    void SetXRange(double xStart, double xEnd) { xIntegrator.SetRange(xStart, xEnd); }
    std::pair<double, double> GetXRange() const { return xIntegrator.GetRange(); }

    /**
     * Set the integration limits.
     * @param yStart
     * @param yEnd
     */
    void SetYRange(double yStart, double yEnd) { yIntegrator.SetRange(yStart, yEnd); }
    std::pair<double, double> GetYRange() const { return yIntegrator.GetRange(); }

    /**
     * Set the integration limits.
     * @param yStart
     * @param yEnd
     */
    void SetRange(double xStart, double xEnd, double yStart, double yEnd) { xIntegrator.SetRange(xStart, xEnd); yIntegrator.SetRange(yStart, yEnd); }

    /**
     * Set the desired precision.
     * @param precision
     */
    void SetPrecision(double precision) { xIntegrator.SetPrecision(precision); yIntegrator.SetPrecision(precision); }
    double GetPrecision() const { return xIntegrator.GetPrecision(); }

    uint32_t SetMinSteps(uint32_t min) { xIntegrator.SetMinSteps(min); return yIntegrator.SetMinSteps(min); }
    uint32_t GetMinSteps() const { return xIntegrator.GetMinSteps(); }
    uint32_t SetMaxSteps(uint32_t max) { xIntegrator.SetMaxSteps(max); return yIntegrator.SetMaxSteps(max); }
    uint32_t GetMaxSteps() const { return xIntegrator.GetMaxSteps(); }

    /**
     * Fix the minimum and maximum number of steps to the same value.
     * @param minAndMax
     * @return
     */
    uint32_t SetSteps(uint32_t minAndMax) { SetMinSteps(minAndMax); return SetMaxSteps(minAndMax); }

    /**
     * Configure the integration algorithm.
     * @param method
     */
    void SetMethod(KMathIntegrator::EIntegrationMethod method) { xIntegrator.SetMethod(method); yIntegrator.SetMethod(method); }
    KMathIntegrator::EIntegrationMethod GetMethod() const { return xIntegrator.GetMethod(); }

    void ThrowExceptions(bool throwExc) { xIntegrator.ThrowExceptions(throwExc); yIntegrator.ThrowExceptions(throwExc); }
    void FallbackOnLowPrecision(bool fallback) { xIntegrator.FallbackOnLowPrecision(fallback); yIntegrator.FallbackOnLowPrecision(fallback); }
    void UseKahanSumming(bool useKahan) { xIntegrator.UseKahanSumming(useKahan); yIntegrator.UseKahanSumming(useKahan); }

    uint32_t NumberOfIterations() const { return fIterations; }
    uint64_t NumberOfSteps() const { return fSteps; }

    /**
     * Perform the integration.
      * @return
     */
    template<class XIntegrandType2D>
    double Integrate(XIntegrandType2D& func);

    /**
     * Perform the integration.
     * @param xStart Lower integration limit.
     * @param xEnd Upper integration limit.
     * @param yStart Lower integration limit.
     * @param yEnd Upper integration limit.
     * @return
     */
    template<class XIntegrandType2D>
    double Integrate(XIntegrandType2D& func, double xStart, double xEnd, double yStart, double yEnd);

protected:
    KMathIntegrator xIntegrator;
    KMathIntegrator yIntegrator;
    uint32_t fIterations;
    uint64_t fSteps;

    template<class XIntegrandType2D>
    struct InnerFunctor {
        InnerFunctor(XIntegrandType2D& func) : fX(0.0), fIntegrand(func) { }
        double operator() (double y)
        {
            return fIntegrand(fX, y);
        }
        double fX;
        XIntegrandType2D& fIntegrand;
    };

    template<class XIntegrandType2D>
    struct OuterFunctor {
        OuterFunctor(XIntegrandType2D& func, KMathIntegrator& integrator, uint32_t& iterationCounter, uint64_t& stepCounter) :
            fInnerFunctor(func),
            fIntegrator(integrator),
            fIterations(iterationCounter),
            fSteps(stepCounter)
        { }
        double operator() (double x)
        {
            fInnerFunctor.fX = x;
            const double result = fIntegrator.Integrate( fInnerFunctor );
            fIterations += fIntegrator.NumberOfIterations();
            fSteps += fIntegrator.NumberOfSteps();
            return result;
        }
        InnerFunctor<XIntegrandType2D> fInnerFunctor;
        KMathIntegrator& fIntegrator;
        uint32_t& fIterations;
        uint64_t& fSteps;
    };

};

inline KMathIntegrator2D::KMathIntegrator2D(double precision, KMathIntegrator::EIntegrationMethod method) :
    xIntegrator(precision, method),
    yIntegrator(precision, method),
    fIterations(0),
    fSteps(0)
{ }

template<class XIntegrandType2D>
inline double KMathIntegrator2D::Integrate(XIntegrandType2D& integrand)
{
    fIterations = fSteps = 0;
    OuterFunctor<XIntegrandType2D> outerFunctor(integrand, yIntegrator, fIterations, fSteps);
    return xIntegrator.Integrate( outerFunctor );
}

template<class XIntegrandType2D>
inline double KMathIntegrator2D::Integrate(XIntegrandType2D& func, double xStart, double xEnd, double yStart, double yEnd)
{
    SetXRange(xStart, xEnd);
    SetYRange(yStart, yEnd);
    return Integrate(func);
}

} /* namespace katrin */

#endif /* KMATHINTEGRATOR2D_H_ */
