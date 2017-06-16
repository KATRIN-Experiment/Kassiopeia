/**
 * @file KMathRegulaFalsi.h
 *
 * @date 23.11.2015
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KOMMON_CORE_UTILITY_KMATHREGULAFALSI_H_
#define KOMMON_CORE_UTILITY_KMATHREGULAFALSI_H_

#include <KNumeric.h>
#include <KException.h>

#include <functional>
#include <utility>
#include <algorithm>

namespace katrin {

enum class KEMathRegulaFalsiMethod {
   Pegasus, AndersonBjoerck, Illinois
};

template<class XFloatT>
class KMathRegulaFalsi {
public:
    KMathRegulaFalsi(XFloatT relError = 1E-5, uint16_t nMax = 1000, KEMathRegulaFalsiMethod method = KEMathRegulaFalsiMethod::AndersonBjoerck) :
        fRelError(relError), fNMax(nMax), fMethod(method) { };

    void SetRelativeError(XFloatT relError) { fRelError = relError; }
    void SetNMax(uint16_t nMax) { fNMax = nMax; }
    void SetMethod(KEMathRegulaFalsiMethod method) { fMethod = method; }
    void SetBisectionRatio(XFloatT bisec = 0.1) { fBisec = std::max(0.0, std::min(1.0, bisec)); }

    template<class XCallableT>
    XFloatT FindIntercept(const XCallableT& callable, XFloatT x1, XFloatT x2);

    uint16_t GetNEvaluations() const { return fNCounter; }

private:
    XFloatT fRelError;
    uint16_t fNCounter = 0;
    uint16_t fNMax;
    KEMathRegulaFalsiMethod fMethod;
    XFloatT fBisec = 0.1;
};

template<class XFloatT>
template<class XCallableT>
inline XFloatT KMathRegulaFalsi<XFloatT>::FindIntercept(const XCallableT& callable, XFloatT x1, XFloatT x2)
{
    XFloatT f1 = callable(x1);
    XFloatT f2 = callable(x2);
    fNCounter = 2;

    if (f1*f2 > 0.0) {
        throw KException() << "KMathRegulaFalsi: No intersect between starting points.";
    }

    XFloatT v = x1 - x2;
    const XFloatT bisectionLength = fBisec * std::abs(v);

    do {
        const XFloatT tol = fRelError * x2;

        // determine, if we should make a bisection step
        const XFloatT bisection = (std::abs(v) > bisectionLength);

        XFloatT deltaX = (bisection) ? 0.5*v : v*f2/(f2-f1);
        if (std::abs(deltaX) < tol)
            deltaX = 0.9 * KNumeric::Sign(v) * tol;

        const XFloatT x3 = x2 + deltaX;
        XFloatT f3 = callable(x3);

        if (f3 == 0.0)
            return x3;

        const XFloatT e = KNumeric::Sign(f2) * f3;
        if (e < 0.0) {
            std::swap(x1, x2);
            std::swap(f1, f2);
        }
        x2 = x3;
        std::swap(f2, f3);
        v = x1 - x2;

        if (std::abs(v) <= tol)
            return ( std::abs(f2) <= std::abs(f1) ) ? x2 : x1;

        // prepare next iteration

        double g = 0.5; // fMethod == EMethod::Illinois

        if (fMethod == KEMathRegulaFalsiMethod::Pegasus) {
            g = f3/(f3+f2);
        }
        else if  (fMethod == KEMathRegulaFalsiMethod::AndersonBjoerck) {
            if (bisection) {
                g = f3/(f3+f2);
            }
            else {
                g = 1.0-f2/f3;
                if (g <= 0.0)
                    g = 0.5;
            }
        }

        f1 *= g;
    }
    while (++fNCounter < fNMax);

    throw KException() << "KMathRegulaFalsi: Unable to find an intercept within "
        << fNMax << " function evaluations.";
}

using KMathRegulaFalsiD = KMathRegulaFalsi<double>;

}

#endif /* KOMMON_CORE_UTILITY_KMATHREGULAFALSI_H_ */
