/**
 * @file KMathGamma.h
 *
 * @date 27.11.2015
 * @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */
#ifndef KOMMON_CORE_UTILITY_KMATHGAMMA_H_
#define KOMMON_CORE_UTILITY_KMATHGAMMA_H_

#include "KConst.h"
#include <complex>
#include <type_traits>

namespace katrin {
namespace math {

/**
 * Complex gamma function using the Lanczos approximation.
 * This implementation typically gives a precision of 15 decimals.
 * @param complex number z
 * @return
 *
 * @see https://en.wikipedia.org/wiki/Lanczos_approximation
 */
template<class XFloatT = double>
std::complex<XFloatT> gamma(std::complex<XFloatT> z)
{
    constexpr int g = 7;
    constexpr std::array<XFloatT, g + 2> p = {{
        0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
        -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
        1.5056327351493116e-7 }};
    constexpr XFloatT pi = KConst::Pi<XFloatT>();

    if ( z.real() < 0.5 ) {
        return pi / (sin(pi*z)*gamma( 1.0-z ) );
    }
    z -= 1.0;

    std::complex<XFloatT> x = p[0];
    for (int i=1; i<g+2; i++) {
        x += p[i] / (z+(XFloatT) i);
    }

    const std::complex<XFloatT> t = z + ((XFloatT) g + 0.5);
    return sqrt(2.0*pi) * pow(t,z+0.5) * exp(-t) * x;
}

/**
 * Real value gamma function.
 * Simple wrapper for the complex version of #gamma
 * @param z
 * @return
 */
template<class XFloatT = double, class = typename std::enable_if<std::is_fundamental<XFloatT>::value>::type>
XFloatT gamma(XFloatT x)
{
    return gamma( std::complex<XFloatT>(x) ).real();
}

} }

#endif /* KOMMON_CORE_UTILITY_KMATHGAMMA_H_ */
