/**
 *  @file KFastMath.h
 *
 *  @date Created on: 09.02.2012
 *  @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KFASTMATH_H_
#define KFASTMATH_H_

#include <cmath>
#include <boost/math/special_functions/pow.hpp>

namespace katrin {

/**
 * Experimental static functions with approximations to CPU intense mathematical functions.
 * Your compiler will most certainly throw warnings when including this file!
 */
class KFastMath {
public:
    template< typename XFloatType >
    static XFloatType Sqrt_Heron(XFloatType a, uint32_t nIterations = 5);

    template< typename XFloatType >
    static XFloatType Exp_Taylor(XFloatType x, int32_t order);

    template< int NIteratons, typename XFloatType >
    static XFloatType Exp_Taylor(XFloatType x);

    static double FastSqrt_Log2(double x);
    static double FastSqrt_Bab1(double x);
    static double FastSqrt_Bab2(double x);

    static float InvSqrt_Q3(float x, uint32_t n = 2);
    static double InvSqrt_Q3(double x, uint32_t n = 2);

    template< typename XFloatType >
    static XFloatType FastSqrt_Q3(XFloatType x, uint32_t n = 2);

    static double FastExp(double y);
};

template< typename XFloatType >
inline XFloatType KFastMath::Sqrt_Heron(XFloatType a, uint32_t nIterations)
{
//    if (a == 1.0)
//        return a;

    double x = a; // Choose the start x to be a (not suitable for large a)
    for (uint32_t i = 0; i < nIterations; i++) {
        x = ( x + a / x ) / 2.0; // Heron formula
    }

    return x;
}


inline double KFastMath::FastSqrt_Log2(const double x)
{
    union {
        int64_t i;
        double x;
    } u;
    u.x = x;
    u.i = (((int64_t)1)<<61) + (u.i >> 1) - (((int64_t)1)<<51);
    return u.x;
}

inline double KFastMath::FastSqrt_Bab1(const double x)
{
    union {
        int64_t i;
        double x;
    } u;
    u.x = x;
    u.i = (((int64_t)1)<<61) + (u.i >> 1) - (((int64_t)1)<<51);
    // One Babylonian Step
    u.x = 0.5F * (u.x + x/u.x);
    return u.x;
}

inline double KFastMath::FastSqrt_Bab2(const double x)
{
    union
    {
        int64_t i;
        double x;
    } u;
    u.x = x;
    u.i = (((int64_t)1)<<61) + (u.i >> 1) - (((int64_t)1)<<51);

    // Two Babylonian Steps (simplified from:)
    // u.x = 0.5F * (u.x + x/u.x);
    // u.x = 0.5F * (u.x + x/u.x);
    u.x =       u.x + x/u.x;
    u.x = 0.25F*u.x + x/u.x;

    return u.x;
}

inline float KFastMath::InvSqrt_Q3(const float x, uint32_t n)
{
    const float xhalf = 0.5F * x;

    union // get bits for floating value
    {
        float x;
        int32_t i;
    } u;
    u.x = x;
    u.i = 0x5f375a86 - (u.i >> 1);  // gives initial guess y0
    for (uint32_t i = 0; i < n; ++i)
        u.x = u.x * (1.5F - xhalf * u.x * u.x);  // Newton step, repeating increases accuracy
    return u.x;
}

inline double KFastMath::InvSqrt_Q3(const double x, uint32_t n)
{
    const double xhalf = 0.5F * x;

    union // get bits for floating value
    {
        double x;
        int64_t i;
    } u;
    u.x = x;
    u.i = 0x5fe6eb50c7b537a9 - (u.i >> 1);  // gives initial guess y0
    for (uint32_t i = 0; i < n; ++i)
        u.x = u.x * (1.5F - xhalf * u.x * u.x);  // Newton step, repeating increases accuracy
    return u.x;
}

template< typename XFloatType >
inline XFloatType KFastMath::FastSqrt_Q3(XFloatType x, uint32_t n)
{
    return x * InvSqrt_Q3(x, n);
}

inline double KFastMath::FastExp(double y)
{
    double d;
    *((int*)(&d) + 0) = 0;
    *((int*)(&d) + 1) = (int)(1512775 * y + 1072632447);
    return d;
}

template< typename XFloatType >
inline XFloatType KFastMath::Exp_Taylor(XFloatType x, int32_t n)
{
    XFloatType sum = 1.0; // initialize sum of series

    for (int32_t i = n - 1; i > 0; --i )
        sum = 1.0 + x * sum / (XFloatType) i;

    return sum;
}

template< int NIteratons, typename XFloatType >
inline XFloatType KFastMath::Exp_Taylor(XFloatType x)
{
    return boost::math::pow<NIteratons>(1.0 + x / (XFloatType) NIteratons );
}

}

#endif /* KFASTMATH_H_ */
