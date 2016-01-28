/*
 * @file   KMathIntegrator.h
 *
 * @date   Created on: 08.09.2012
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KMATHINTEGRATOR_H_
#define KMATHINTEGRATOR_H_

#include "KException.h"
#include "KMathKahanSum.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <utility>

namespace katrin
{

/**
 * Integration methods.
 */
enum class KEMathIntegrationMethod {
    Trapezoidal, Simpson, K3, K4, Romberg
};

// forward declarations

template<class XSamplingPolicy>
class KMathIntegratorPrototype;

namespace integrator_policies {
    struct PlainSumming;
    struct KahanSumming;
}

/**
 * Type definition for a numerical integrator using Kahan summing.
 */
using KMathIntegrator = KMathIntegratorPrototype<integrator_policies::KahanSumming>;

/**
 * A numerical integrator, implementing a trapezoidal, the simpson and the romberg method.
 * The actual algorithms are inspired by "Numerical Recipes 3rd Edition".
 * The intergrator is configured by specifying the integration limits, the minimum and maximum number of
 * integration steps (rounded to a power of 2) and the desired precision.
 * The number of actual integration steps will be automatically increased until the precision is achieved.
 * If a given precision is not achieved within the maximum number of steps, either a warning is printed
 * or an exception is thrown.
 */
template<class XSamplingPolicy = integrator_policies::PlainSumming>
class KMathIntegratorPrototype : private XSamplingPolicy
{
public:
    KMathIntegratorPrototype(double precision = 1E-4, KEMathIntegrationMethod method = KEMathIntegrationMethod::Simpson);
    virtual ~KMathIntegratorPrototype() { }

    /**
     * Perform the integration.
      * @return
     */
    template<class XIntegrandType>
    double Integrate(XIntegrandType& func);

    /**
     * Perform the integration.
     * @param xStart Lower integration limit.
     * @param xEnd Upper integration limit.
      * @return
     */
    template<class XIntegrandType>
    double Integrate(XIntegrandType& func, double xStart, double xEnd);

    /**
     * Set the integration limits.
     * @param xStart
     * @param xEnd
     */
    void SetRange(double xStart, double xEnd) { fXStart = xStart; fXEnd = xEnd; }
    std::pair<double, double> GetRange() const { return std::make_pair(fXStart, fXEnd); }

    /**
     * Set the desired precision.
     * @param precision
     */
    void SetPrecision(double precision);
    double GetPrecision() const { return fPrecision; }

    uint32_t SetMinSteps(uint32_t min);
    uint32_t GetMinSteps() const { return 1 << fJMin; }
    uint32_t SetMaxSteps(uint32_t max);
    uint32_t GetMaxSteps() const { return 1 << fJMax; }

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
    void SetMethod(KEMathIntegrationMethod method) { fMethod = method; }
    KEMathIntegrationMethod GetMethod() const { return fMethod; }

    void ThrowExceptions(bool throwExc) { fThrowExceptions = throwExc; }
    void FallbackOnLowPrecision(bool fallback) { fFallbackOnLowPrec = fallback; }

    uint32_t NumberOfIterations() const { return fIteration; }
    uint64_t NumberOfSteps() const { return (fIteration == 0) ? 0 : ((uint64_t) 1 << (fIteration - 1)) + 1; }

protected:
    static uint32_t ilog2_ceil(uint32_t v);

    struct K1DPolyInterpolator;

    void Reset();

    template<class XIntegrandType>
    double NextTrapezoid(XIntegrandType& integrand);
    template<class XIntegrandType>
    double QTrap(XIntegrandType& integrand);
    template<class XIntegrandType>
    double QSimp(XIntegrandType& integrand);
    template<class XIntegrandType>
    double QRomb(XIntegrandType& integrand, const uint32_t K = 5);

    double fXStart;
    double fXEnd;

    double fPrecision;
    uint32_t fJMin, fJMax;    // 2^(fJMax) is the maximum, 2^(fJMin) the minimum allowed number of steps.

    KEMathIntegrationMethod fMethod;

    double fCurrentResult;
    uint32_t fIteration;

    bool fThrowExceptions;
    bool fFallbackOnLowPrec;
};

class KMathIntegratorException: public KExceptionPrototype<KMathIntegratorException, KException> { };

template<class XSamplingPolicy>
struct KMathIntegratorPrototype<XSamplingPolicy>::K1DPolyInterpolator {
    K1DPolyInterpolator(std::vector<double> &xv, std::vector<double> &yv, int32_t m) :
        n(xv.size()), mm(m), jsav(0), cor(0), xx(&xv[0]), yy(&yv[0]), dy(0.0) {
        dj = std::min(1, (int) std::pow((double) n, 0.25));
    }

    double RawInterpolate(int32_t jlo, double x);

    int32_t n, mm, jsav, cor, dj;
    const double *xx, *yy;
    double dy;
};

template<class XSamplingPolicy>
inline uint32_t KMathIntegratorPrototype<XSamplingPolicy>::ilog2_ceil(uint32_t v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    uint32_t r = 0;
    while (v >>= 1)
        ++r;
    return r;
}

template<class XSamplingPolicy>
template<class XIntegrandType>
inline double KMathIntegratorPrototype<XSamplingPolicy>::NextTrapezoid(XIntegrandType& integrand) {

    ++fIteration;
    if (fIteration == 1) {
        const double sum = XSamplingPolicy::SumSamplingPoints(2, fXStart, (fXEnd - fXStart), integrand);
//        const double sum = integrand(fXStart) + integrand(fXEnd);
        fCurrentResult = 0.5 * (fXEnd - fXStart) * sum;
    }
    else {
        uint32_t it = 1;
        for (uint32_t j = 1; j < fIteration - 1; j++)
            it <<= 1;

        const double del = (fXEnd - fXStart) / (double) it;
        const double xStart = fXStart + 0.5 * del;

        const double sum = XSamplingPolicy::SumSamplingPoints(it, xStart, del, integrand);

        fCurrentResult = 0.5 * (fCurrentResult + del * sum);
    }
    return fCurrentResult;
}

template<class XSamplingPolicy>
inline KMathIntegratorPrototype<XSamplingPolicy>::KMathIntegratorPrototype(double precision, KEMathIntegrationMethod method) :
	fXStart(0.0),
	fXEnd(0.0),
	fPrecision(1.0),
	fJMin(2),
	fJMax(16),
	fMethod(method),
	fCurrentResult(0.0),
	fIteration(0),
	fThrowExceptions(false),
	fFallbackOnLowPrec(false)
{
	SetPrecision(precision);
}

template<class XSamplingPolicy>
inline void KMathIntegratorPrototype<XSamplingPolicy>::Reset()
{
    fIteration = 0;
    fCurrentResult = 0.0;
}

template<class XSamplingPolicy>
inline void KMathIntegratorPrototype<XSamplingPolicy>::SetPrecision(double precision)
{
	if (precision <= 0.0)
		precision = 1.0;
	fPrecision = precision;
}

template<class XSamplingPolicy>
inline uint32_t KMathIntegratorPrototype<XSamplingPolicy>::SetMinSteps(uint32_t min)
{
    fJMin = ilog2_ceil(min);
    if (fJMax < fJMin)
        fJMax = fJMin;
    return GetMinSteps();
}

template<class XSamplingPolicy>
inline uint32_t KMathIntegratorPrototype<XSamplingPolicy>::SetMaxSteps(uint32_t max)
{
    fJMax = ilog2_ceil(max);
    if (fJMin > fJMax)
        fJMin = fJMax;
    return GetMaxSteps();
}

template<class XSamplingPolicy>
template<class XIntegrandType>
inline double KMathIntegratorPrototype<XSamplingPolicy>::Integrate(XIntegrandType& integrand)
{
    Reset();

    if( fJMax < fJMin )
        fJMax = fJMin;

    switch( fMethod ) {
        case KEMathIntegrationMethod::Trapezoidal :
            return QTrap(integrand);
        case KEMathIntegrationMethod::Simpson :
            return QSimp(integrand);
        case KEMathIntegrationMethod::K3 :
            return QRomb(integrand, 3 );
        case KEMathIntegrationMethod::K4 :
            return QRomb(integrand, 4 );
        case KEMathIntegrationMethod::Romberg :
            return QRomb(integrand, 5 );
        default :
            throw KMathIntegratorException() << "Invalid integration method specified.";
    }
}

template<class XSamplingPolicy>
template<class XIntegrandType>
inline double KMathIntegratorPrototype<XSamplingPolicy>::Integrate(XIntegrandType& integrand, double xStart, double xEnd)
{
    SetRange(xStart, xEnd);
    return Integrate(integrand);
}

template<class XSamplingPolicy>
template<class XIntegrandType>
inline double KMathIntegratorPrototype<XSamplingPolicy>::QTrap(XIntegrandType& integrand)
{
    /*Returns the integral of the function or functor func from a to b. The constants EPS can be
     set to the desired fractional accuracy and JMAX so that 2 to the power JMAX-1 is the maximum
     allowed number of steps. Integration is performed by the trapezoidal rule.*/
    double s = 0.0, olds = 0.0; //Initial value of olds is arbitrary.
    for (uint32_t j = 0; j <= fJMax; j++) {
        olds = s;
        s = NextTrapezoid(integrand);
        if (j >= fJMin) //Avoid spurious early convergence.
            if (fabs(s - olds) < fPrecision * fabs(olds) || (s == 0.0 && olds == 0.0))
                return s;
    }

    if (fThrowExceptions) {
        if ( std::isnormal(s) )
            throw KMathIntegratorException() << "Precision insufficient in routine QTrap.";
        else
            throw KMathIntegratorException() << "Non finite result in routine QTrap.";
    }

    return s;
}

template<class XSamplingPolicy>
template<class XIntegrandType>
inline double KMathIntegratorPrototype<XSamplingPolicy>::QSimp(XIntegrandType& integrand)
{
    /*Returns the integral of the function or functor func from a to b. The constants EPS can be
     set to the desired fractional accuracy and JMAX so that 2 to the power JMAX-1 is the maximum
     allowed number of steps. Integration is performed by Simpson’s rule.*/
    double s = 0.0, st = 0.0, ost = 0.0, os = 0.0;
    for (uint32_t j = 0; j <= fJMax; j++) {
        ost = st;
        os = s;
        st = NextTrapezoid(integrand);
        s = (4.0 * st - ost) / 3.0; //Compare equation (4.2.4), above.
        if (j >= fJMin) //Avoid spurious early convergence.
            if (fabs(s - os) < fPrecision * fabs(os) || (s == 0.0 && os == 0.0))
                return s;
    }


    if ( std::isnormal(s) ) {
        if (fThrowExceptions)
            throw KMathIntegratorException() << "Precision insufficient in routine QSimp.";

        if (fFallbackOnLowPrec) {
            Reset();
            return QTrap(integrand);
        }
        else
            return s;
    }
    else {
        if (fThrowExceptions)
            throw KMathIntegratorException() << "Non finite result in routine QSimp.";

        Reset();
        return QTrap(integrand);
    }
}

template<class XSamplingPolicy>
template<class XIntegrandType>
inline double KMathIntegratorPrototype<XSamplingPolicy>::QRomb(XIntegrandType& integrand, const uint32_t K)
{
    /*Returns the integral of the function or functor func from a to b. Integration is performed by
     Romberg’s method of order 2K, where, e.g., K=2 is Simpson’s rule.*/

    /*Here EPS is the fractional accuracy desired, as determined by the extrapolation error es-
     timate; JMAX limits the total number of steps; K is the number of points used in the
     extrapolation.*/

    const uint32_t jMax = std::max(fJMax+1, K);

    std::vector<double> s(jMax, 0.0), h(jMax+1, 0.0); // These store the successive trapezoidal approximations and their relative stepsizes.
    K1DPolyInterpolator polint(h, s, K);

    double ss = 0.0/*, oss = 0.0*/;

    h[0] = 1.0;
    for (uint32_t j = 1; j <= jMax; j++) {
        s[j - 1] = NextTrapezoid(integrand);
        if (j >= K && j > fJMin) {
//            oss = ss;
            ss = polint.RawInterpolate(j - K, 0.0);

            if (fabs(polint.dy) <= fPrecision * fabs(ss))
                return ss;
        }
        h[j] = 0.25 * h[j - 1];
        /*This is a key step: The factor is 0.25 even though the stepsize is decreased by only
         0.5. This makes the extrapolation a polynomial in h2 as allowed by equation (4.2.1),
         not just a polynomial in h.*/
    }

    if ( std::isnormal(ss) ) {
        if (fThrowExceptions)
            throw KMathIntegratorException() << "Precision insufficient in routine QRomb.";

        if (fFallbackOnLowPrec) {
            Reset();
            return QSimp(integrand);
        }
        else
            return ss;
    }
    else {
        if (fThrowExceptions)
            throw KMathIntegratorException() << "Non finite result in routine QRomb.";

        Reset();
        return QSimp(integrand);
    }
}

template<class XSamplingPolicy>
inline double KMathIntegratorPrototype<XSamplingPolicy>::K1DPolyInterpolator::RawInterpolate(int32_t jl, double x)
{
    /*Given a value x, and using pointers to data xx and yy, this routine returns an interpolated
     value y, and stores an error estimate dy. The returned value is obtained by mm-point polynomial
     interpolation on the subrange xx[jl..jl+mm-1].*/
    int32_t i, m, ns = 0;
    double y, den, dif, dift, ho, hp, w;
    const double *xa = &xx[jl], *ya = &yy[jl];
    std::vector<double> c(mm), d(mm);
    dif = fabs(x - xa[0]);
    for (i = 0; i < mm; i++) { // Here we find the index ns of the closest table entry,
        if ((dift = fabs(x - xa[i])) < dif) {
            ns = i;
            dif = dift;
        }
        c[i] = ya[i]; // and initialize the tableau of c’s and d’s.
        d[i] = ya[i];
    }
    y = ya[ns--]; // This is the initial approximation to y.
// For each column of the tableau, we loop over the current c’s and d’s and update them.
    for (m = 1; m < mm; m++) {
        for (i = 0; i < mm - m; i++) {
            ho = xa[i] - x;
            hp = xa[i + m] - x;
            w = c[i + 1] - d[i];
            if ((den = ho - hp) == 0.0)
                throw KMathIntegratorException() << "Poly_interp error"; // This error can occur only if two input xa’s are (to within roundoff) identical.
            den = w / den;
            d[i] = hp * den; // Here the c’s and d’s are updated.
            c[i] = ho * den;
        }
        y += (dy = (2 * (ns + 1) < (mm - m) ? c[ns + 1] : d[ns--]));
        /*After each column in the tableau is completed, we decide which correction, c or d, we
         want to add to our accumulating value of y, i.e., which path to take through the tableau
         — forking up or down. We do this in such a way as to take the most “straight line”
         route through the tableau to its apex, updating ns accordingly to keep track of where
         we are. This route keeps the partial approximations centered (insofar as possible) on
         the target x. The last dy added is thus the error indication.*/
    }
    return y;
}

namespace integrator_policies {

struct PlainSumming
{
    template<class XIntegrandType>
    double SumSamplingPoints(uint32_t n, const double& xStart, const double& del, XIntegrandType& integrand) const
    {
        double sum = 0.0;
        for (uint32_t j = 0; j < n; j++) {
            const double x = xStart + (double) j * del;
            sum += integrand(x);
        }
        return sum;
    }
};

struct KahanSumming
{
    template<class XIntegrandType>
    double SumSamplingPoints(uint32_t n, const double& xStart, const double& del, XIntegrandType& integrand) const
    {
        KMathKahanSum sum;
        for (uint32_t j = 0; j < n; j++) {
            const double x = xStart + (double) j * del;
            sum += integrand(x);
        }
        return sum;
    }
};

}

} /* namespace katrin */
#endif /* KMATHINTEGRATOR_H_ */
