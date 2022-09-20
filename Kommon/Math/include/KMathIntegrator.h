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

#include <algorithm>
#include <functional>
#include <cmath>
#include <map>
#include <utility>
#include <vector>

#ifdef KASPER_USE_GSL
#include <gsl/gsl_integration.h>
#include "KGslErrorHandler.h"

namespace {

//inspired by fitrium - thx!
using FType = const std::function<double(double)>;
/**
 * Create a gsl_function from c++ template function
 *
 * @param integrand The c++ style function object
 * @returns A gsl_function usable by the GSL integration functions
 */
inline gsl_function GSLFunction(FType& integrand)
{
	gsl_function gsl_func;
	gsl_func.function = [](double x, void* param) { return (*static_cast<FType*>(param))(x); };
	gsl_func.params = (void*)&integrand;
	return gsl_func;

}
}
#endif /* KASPER_USE_GSL */


namespace katrin
{

/**
 * Integration methods.
 */
enum class KEMathIntegrationMethod
{
    Trapezoidal,
    Simpson,
    K3,
    K4,
    Romberg,
	QAGS
};

// forward declarations
namespace policies
{
struct PlainSumming;
struct KahanSumming;
}  // namespace policies

/**
 * A numerical integrator, implementing a trapezoidal, the simpson and the romberg method.
 * The actual algorithms are inspired by "Numerical Recipes 3rd Edition".
 * The intergrator is configured by specifying the integration limits, the minimum and maximum number of
 * integration steps (rounded to a power of 2) and the desired precision.
 * The number of actual integration steps will be automatically increased until the precision is achieved.
 * If a given precision is not achieved within the maximum number of steps, either a warning is printed
 * or an exception is thrown.
 */
template<class XFloatT = double, class XSamplingPolicy = policies::PlainSumming>
class KMathIntegrator : private XSamplingPolicy
{
  public:
    KMathIntegrator(XFloatT precision = 1E-4, KEMathIntegrationMethod method = KEMathIntegrationMethod::Simpson);
    virtual ~KMathIntegrator() = default;

    /**
     * Perform the integration.
      * @return
     */
    template<class XIntegrandType> XFloatT Integrate(XIntegrandType&& func);

    /**
     * Perform the integration.
     * @param xStart Lower integration limit.
     * @param xEnd Upper integration limit.
      * @return
     */
    template<class XIntegrandType> XFloatT Integrate(XIntegrandType&& func, XFloatT xStart, XFloatT xEnd);

    /**
     * Set the integration limits.
     * @param xStart
     * @param xEnd
     */
    void SetRange(XFloatT xStart, XFloatT xEnd)
    {
        fXStart = xStart;
        fXEnd = xEnd;
    }
    std::pair<XFloatT, XFloatT> GetRange() const
    {
        return std::make_pair(fXStart, fXEnd);
    }

    /**
     * Set the desired precision.
     * @param precision
     */
    void SetPrecision(XFloatT precision);
    XFloatT GetPrecision() const
    {
        return fPrecision;
    }

    uint32_t SetMinSteps(int32_t min);
    uint32_t GetMinSteps() const
    {
        return (1 << fJMin) + 1;
    }
    uint32_t SetMaxSteps(int32_t max);
    uint32_t GetMaxSteps() const
    {
        return (1 << fJMax) + 1;
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
        fMethod = method;
    }
    KEMathIntegrationMethod GetMethod() const
    {
        return fMethod;
    }

    void ThrowExceptions(bool throwExc)
    {
        fThrowExceptions = throwExc;
    }
    void FallbackOnLowPrecision(bool fallback)
    {
        fFallbackOnLowPrec = fallback;
    }

    uint32_t NumberOfIterations() const
    {
        return fIteration;
    }
    uint32_t NumberOfSteps() const;

  protected:
    static uint32_t ilog2_ceil(uint32_t v);
    static uint32_t fourthRoot(uint32_t A);

    struct K1DPolyInterpolator;

    void Reset();

    template<class XIntegrandType> XFloatT NextTrapezoid(XIntegrandType&& integrand);
    template<class XIntegrandType> XFloatT QTrap(XIntegrandType&& integrand);
    template<class XIntegrandType> XFloatT QSimp(XIntegrandType&& integrand);
    template<class XIntegrandType> XFloatT QRomb(XIntegrandType&& integrand, const uint32_t K = 5);
#ifdef KASPER_USE_GSL
    template<class XIntegrandType> XFloatT QAGIU(XIntegrandType&& integrand);
    template<class XIntegrandType> XFloatT QAGS(XIntegrandType&& integrand);
#endif

    XFloatT fXStart;
    XFloatT fXEnd;

    XFloatT fPrecision;
    uint32_t fJMin, fJMax;  // 2^(fJMax) is the maximum, 2^(fJMin) the minimum allowed number of steps.

    KEMathIntegrationMethod fMethod;

    XFloatT fCurrentResult;
    uint32_t fIteration;

    bool fThrowExceptions;
    bool fFallbackOnLowPrec;
};

class KMathIntegratorException : public KExceptionPrototype<KMathIntegratorException, KException>
{};

template<class XFloatT, class XSamplingPolicy> struct KMathIntegrator<XFloatT, XSamplingPolicy>::K1DPolyInterpolator
{
    K1DPolyInterpolator(std::vector<XFloatT>& xv, std::vector<XFloatT>& yv, int32_t m) :
        n(xv.size()),
        mm(m),
        jsav(0),
        cor(0),
        xx(&xv[0]),
        yy(&yv[0]),
        dy(0.0)
    {
        dj = std::max<int32_t>(1, fourthRoot(n));
    }

    XFloatT RawInterpolate(int32_t jlo, XFloatT x);

    int32_t n, mm, jsav, cor, dj;
    const XFloatT *xx, *yy;
    XFloatT dy;
};

template<class XFloatT, class XSamplingPolicy>
inline uint32_t KMathIntegrator<XFloatT, XSamplingPolicy>::ilog2_ceil(uint32_t v)
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

template<class XFloatT, class XSamplingPolicy>
inline uint32_t KMathIntegrator<XFloatT, XSamplingPolicy>::fourthRoot(uint32_t A)
{
    uint32_t x = 1;
    while (((x * x) * (x * x)) <= A) {
        ++x;
    }
    return --x;
    //    return std::max(0, (int) std::pow((XFloatT) x, 1.0/4.0));
}

template<class XFloatT, class XSamplingPolicy>
template<class XIntegrandType>
inline XFloatT KMathIntegrator<XFloatT, XSamplingPolicy>::NextTrapezoid(XIntegrandType&& integrand)
{

    ++fIteration;
    if (fIteration == 1) {
        const XFloatT sum = XSamplingPolicy::SumSamplingPoints(2, fXStart, (fXEnd - fXStart), integrand);
        //        const XFloatT sum = integrand(fXStart) + integrand(fXEnd);
        fCurrentResult = 0.5 * (fXEnd - fXStart) * sum;
    }
    else {
        uint32_t it = 1;
        for (uint32_t j = 1; j < fIteration - 1; j++)
            it <<= 1;

        const XFloatT del = (fXEnd - fXStart) / (XFloatT) it;
        const XFloatT xStart = fXStart + 0.5 * del;

        const XFloatT sum = XSamplingPolicy::SumSamplingPoints(it, xStart, del, integrand);

        fCurrentResult = 0.5 * (fCurrentResult + del * sum);
    }
    return fCurrentResult;
}

template<class XFloatT, class XSamplingPolicy>
inline KMathIntegrator<XFloatT, XSamplingPolicy>::KMathIntegrator(XFloatT precision, KEMathIntegrationMethod method) :
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

template<class XFloatT, class XSamplingPolicy> inline void KMathIntegrator<XFloatT, XSamplingPolicy>::Reset()
{
    fIteration = 0;
    fCurrentResult = 0.0;
}

template<class XFloatT, class XSamplingPolicy>
inline void KMathIntegrator<XFloatT, XSamplingPolicy>::SetPrecision(XFloatT precision)
{
    if (precision <= 0.0)
        precision = 1.0;
    fPrecision = precision;
}

template<class XFloatT, class XSamplingPolicy>
inline uint32_t KMathIntegrator<XFloatT, XSamplingPolicy>::SetMinSteps(int32_t min)
{
    if (min < 2)
        fJMin = 2;
    else
        fJMin = ilog2_ceil(min - 1);

    if (fJMax < fJMin)
        fJMax = fJMin;

    return GetMinSteps();
}

template<class XFloatT, class XSamplingPolicy>
inline uint32_t KMathIntegrator<XFloatT, XSamplingPolicy>::SetMaxSteps(int32_t max)
{
    if (max < 2)
        fJMax = 16;
    else
        fJMax = ilog2_ceil(max - 1);

    if (fJMin > fJMax)
        fJMin = fJMax;

    return GetMaxSteps();
}

template<class XFloatT, class XSamplingPolicy>
template<class XIntegrandType>
inline XFloatT KMathIntegrator<XFloatT, XSamplingPolicy>::Integrate(XIntegrandType&& integrand)
{
    Reset();

    if (fJMax < fJMin)
        fJMax = fJMin;

    if ((std::isinf(fXEnd) and fXEnd>0))
#ifdef KASPER_USE_GSL
    	return QAGIU(std::forward<XIntegrandType>(integrand));
#else
    	throw KMathIntegratorException() << "Cannot integrate up to infinity without GSL libraries.";
#endif

    switch (fMethod) {
        case KEMathIntegrationMethod::Trapezoidal:
            return QTrap(std::forward<XIntegrandType>(integrand));
        case KEMathIntegrationMethod::Simpson:
            return QSimp(std::forward<XIntegrandType>(integrand));
        case KEMathIntegrationMethod::K3:
            return QRomb(std::forward<XIntegrandType>(integrand), 3);
        case KEMathIntegrationMethod::K4:
            return QRomb(std::forward<XIntegrandType>(integrand), 4);
        case KEMathIntegrationMethod::Romberg:
            return QRomb(std::forward<XIntegrandType>(integrand), 5);
        case KEMathIntegrationMethod::QAGS:
#ifdef KASPER_USE_GSL
        	return QAGS(std::forward<XIntegrandType>(integrand));
#else
            throw KMathIntegratorException() << "Cannot use QAGS/GSL in build without GSL libraries.";
#endif
        default:
            throw KMathIntegratorException() << "Invalid integration method specified.";
    }
}

template<class XFloatT, class XSamplingPolicy>
template<class XIntegrandType>
inline XFloatT KMathIntegrator<XFloatT, XSamplingPolicy>::Integrate(XIntegrandType&& integrand, XFloatT xStart,
                                                                    XFloatT xEnd)
{
    SetRange(xStart, xEnd);
    return Integrate(std::forward<XIntegrandType>(integrand));
}

template<class XFloatT, class XSamplingPolicy>
template<class XIntegrandType>
inline XFloatT KMathIntegrator<XFloatT, XSamplingPolicy>::QTrap(XIntegrandType&& integrand)
{
    /*Returns the integral of the function or functor func from a to b. The constants EPS can be
     set to the desired fractional accuracy and JMAX so that 2 to the power JMAX-1 is the maximum
     allowed number of steps. Integration is performed by the trapezoidal rule.*/
    XFloatT s = 0.0, olds = 0.0;  //Initial value of olds is arbitrary.
    for (uint32_t j = 0; j <= fJMax; j++) {
        olds = s;
        s = NextTrapezoid(integrand);
        if (j >= fJMin)  //Avoid spurious early convergence.
            if (fabs(s - olds) < fPrecision * fabs(olds) || (s == 0.0 && olds == 0.0))
                return s;
    }

    if (fThrowExceptions) {
        if (std::isnormal(s))
            throw KMathIntegratorException() << "Precision insufficient in routine QTrap.";
        else
            throw KMathIntegratorException() << "Non finite result in routine QTrap.";
    }

    return s;
}

template<class XFloatT, class XSamplingPolicy>
template<class XIntegrandType>
inline XFloatT KMathIntegrator<XFloatT, XSamplingPolicy>::QSimp(XIntegrandType&& integrand)
{
    /*Returns the integral of the function or functor func from a to b. The constants EPS can be
     set to the desired fractional accuracy and JMAX so that 2 to the power JMAX-1 is the maximum
     allowed number of steps. Integration is performed by Simpson’s rule.*/
    XFloatT s = 0.0, st = 0.0, ost = 0.0, os = 0.0;
    for (uint32_t j = 0; j <= fJMax; j++) {
        ost = st;
        os = s;
        st = NextTrapezoid(integrand);
        s = (4.0 * st - ost) / 3.0;  //Compare equation (4.2.4), above.
        if (j >= fJMin)              //Avoid spurious early convergence.
            if (fabs(s - os) < fPrecision * fabs(os) || (s == 0.0 && os == 0.0))
                return s;
    }


    if (std::isnormal(s)) {
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

template<class XFloatT, class XSamplingPolicy>
template<class XIntegrandType>
inline XFloatT KMathIntegrator<XFloatT, XSamplingPolicy>::QRomb(XIntegrandType&& integrand, const uint32_t K)
{
    /*Returns the integral of the function or functor func from a to b. Integration is performed by
     Romberg’s method of order 2K, where, e.g., K=2 is Simpson’s rule.*/

    /*Here EPS is the fractional accuracy desired, as determined by the extrapolation error es-
     timate; JMAX limits the total number of steps; K is the number of points used in the
     extrapolation.*/

    const uint32_t jMax = std::max(fJMax + 1, K);

    std::vector<XFloatT> s(jMax, 0.0),
        h(jMax + 1, 0.0);  // These store the successive trapezoidal approximations and their relative stepsizes.
    K1DPolyInterpolator polint(h, s, K);

    XFloatT ss = 0.0 /*, oss = 0.0*/;

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

    if (std::isnormal(ss)) {
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

#ifdef KASPER_USE_GSL
template<class XFloatT, class XSamplingPolicy>
template<class XIntegrandType>
inline XFloatT KMathIntegrator<XFloatT, XSamplingPolicy>::QAGS(XIntegrandType&& integrand)
{
    /*Use gsl implementation for integral int_a^b. Implementation is experimental. */
	const uint ws_size = 100;
    gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(ws_size);
    FType proxy = integrand;
    gsl_function F=GSLFunction(proxy);

    XFloatT AbsError;

    try {
    	gsl_integration_qags(&F,fXStart,fXEnd,0,fPrecision,ws_size,workspace,&fCurrentResult,&AbsError);
    }
    catch (KGslException &e) {
        if (fThrowExceptions)
        	 throw KMathIntegratorException() << "Error encountered in routine QAGS: " << e.what();
    }

//    KGslErrorHandler::GetInstance().Reset();

    fIteration+=ilog2_ceil(workspace->size);	//crude approximation, since QAGS log2(workspace->size) is non integer
    gsl_integration_workspace_free(workspace);

    if (fabs(AbsError/fCurrentResult)<fPrecision)
        return fCurrentResult;

    if (std::isnormal(fCurrentResult)) {
		if (fFallbackOnLowPrec) {
			Reset();
			return QRomb(std::forward<XIntegrandType>(integrand), 5);
		}
		else
			return fCurrentResult;
    }
	else {
		if (fThrowExceptions)
			throw KMathIntegratorException() << "Non finite result in routine QAGS.";

		Reset();
		return QRomb(std::forward<XIntegrandType>(integrand), 5);
	}
}

template<class XFloatT, class XSamplingPolicy>
template<class XIntegrandType>
inline XFloatT KMathIntegrator<XFloatT, XSamplingPolicy>::QAGIU(XIntegrandType&& integrand)
{
    /*Use gsl implementation for integral int_xmin^infty. Use case is the krypton spectrum. Implementation is experimental. */
	const uint ws_size = 100;
    gsl_integration_workspace* workspace = gsl_integration_workspace_alloc(ws_size);
    FType proxy = integrand;
    gsl_function F=GSLFunction(proxy);

    double Precision = fPrecision<1E-5 ? 1E-5 : fPrecision;
    // QAGIU is only used far above the last krypton line with very small rate. Too large precision does not work here.

    XFloatT AbsError;

    try {
    	gsl_integration_qagiu(&F,fXStart,0,Precision,ws_size,workspace,&fCurrentResult,&AbsError);
    }
    catch (KGslException &e) {
        if (fThrowExceptions)
        	 throw KMathIntegratorException() << "Error encountered in routine QAGIU: " << e.what();
    }

//    KGslErrorHandler::GetInstance().Reset();

    fIteration+=ilog2_ceil(workspace->size);	//crude approximation, since QAGIU log2(workspace->size) is non integer
    gsl_integration_workspace_free(workspace);

    if (std::isnormal(fCurrentResult) or fCurrentResult==0.0)
    	return fCurrentResult;
    else
        throw KMathIntegratorException() << "Non finite result in routine QAGIU.";
}
#endif /* KASPER_USE_GSL */

template<class XFloatT, class XSamplingPolicy>
inline XFloatT KMathIntegrator<XFloatT, XSamplingPolicy>::K1DPolyInterpolator::RawInterpolate(int32_t jl, XFloatT x)
{
    /*Given a value x, and using pointers to data xx and yy, this routine returns an interpolated
     value y, and stores an error estimate dy. The returned value is obtained by mm-point polynomial
     interpolation on the subrange xx[jl..jl+mm-1].*/
    int32_t i, m, ns = 0;
    XFloatT y, den, dif, dift, ho, hp, w;
    const XFloatT *xa = &xx[jl], *ya = &yy[jl];
    std::vector<XFloatT> c(mm), d(mm);
    dif = fabs(x - xa[0]);
    for (i = 0; i < mm; i++) {  // Here we find the index ns of the closest table entry,
        if ((dift = fabs(x - xa[i])) < dif) {
            ns = i;
            dif = dift;
        }
        c[i] = ya[i];  // and initialize the tableau of c’s and d’s.
        d[i] = ya[i];
    }
    y = ya[ns--];  // This is the initial approximation to y.
                   // For each column of the tableau, we loop over the current c’s and d’s and update them.
    for (m = 1; m < mm; m++) {
        for (i = 0; i < mm - m; i++) {
            ho = xa[i] - x;
            hp = xa[i + m] - x;
            w = c[i + 1] - d[i];
            if ((den = ho - hp) == 0.0)
                throw KMathIntegratorException()
                    << "Poly_interp error";  // This error can occur only if two input xa’s are (to within roundoff) identical.
            den = w / den;
            d[i] = hp * den;  // Here the c’s and d’s are updated.
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

template<class XFloatT, class XSamplingPolicy>
inline uint32_t KMathIntegrator<XFloatT, XSamplingPolicy>::NumberOfSteps() const
{
    if (fIteration == 0)
        return 0;
    else if (fIteration == 1)
        return 2;
    else
        return (1 << (fIteration - 1)) + 1;
}

namespace policies
{

struct PlainSumming
{
    template<class Float, class XIntegrandType, class Sum = Float>
    static Float SumSamplingPoints(uint32_t n, Float xStart, Float del, XIntegrandType&& integrand)
    {
        Sum sum = 0.0;
        for (uint32_t j = 0; j < n; ++j)
            sum += integrand(xStart + j * del);
        return sum;
    }
};

struct KahanSumming
{
    template<class Float, class XIntegrandType>
    static Float SumSamplingPoints(uint32_t n, Float xStart, Float del, XIntegrandType&& integrand)
    {
        return PlainSumming::SumSamplingPoints<Float, XIntegrandType, KMathKahanSum<Float>>(
            n,
            xStart,
            del,
            std::forward<XIntegrandType>(integrand));
    }
};

}  // namespace policies

} /* namespace katrin */
#endif /* KMATHINTEGRATOR_H_ */
