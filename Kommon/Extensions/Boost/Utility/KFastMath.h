/**
 *  @file KFastMath.h
 *
 *  @date Created on: 09.02.2012
 *  @author Marco Kleesiek <marco.kleesiek@kit.edu>
 */

#ifndef KFASTMATH_H_
#define KFASTMATH_H_

#include <Rtypes.h>
#include <cmath>
#include <boost/math/special_functions/factorials.hpp>

namespace katrin {

/**
 * Experimental static functions with approximations to CPU intense mathematical functions.
 * Your compiler will most certainly throw warnings when including this file!
 */
class KFastMath {
public:

    inline static double FastSqrt_Log2(const double x);
    inline static double FastSqrt_Bab1(const double x);
    inline static double FastSqrt_Bab2(const double x);

    inline static Float_t InvSqrt_Q3(const Float_t x, uint32_t n = 2);
    inline static double InvSqrt_Q3(const double x, uint32_t n = 2);
    inline static Float_t FastSqrt_Q3(const Float_t x, uint32_t n = 2);
    inline static double FastSqrt_Q3(const double x, uint32_t n = 2);

    inline static double FastExp(double y);

    inline static double FastExp_Taylor(double x, int32_t order);

};


inline double KFastMath::FastSqrt_Log2(const double x)
{
    union {
        Long_t i;
        double x;
    } u;
    u.x = x;
    u.i = (((Long_t)1)<<61) + (u.i >> 1) - (((Long_t)1)<<51);
    return u.x;
}


inline double KFastMath::FastSqrt_Bab1(const double x)
{
    union {
        Long_t i;
        double x;
    } u;
    u.x = x;
    u.i = (((Long_t)1)<<61) + (u.i >> 1) - (((Long_t)1)<<51);
    // One Babylonian Step
    u.x = 0.5F * (u.x + x/u.x);
    return u.x;
}

inline double KFastMath::FastSqrt_Bab2(const double x)
{
    union
    {
        Long_t i;
        double x;
    } u;
    u.x = x;
    u.i = (((Long_t)1)<<61) + (u.i >> 1) - (((Long_t)1)<<51);

    // Two Babylonian Steps (simplified from:)
    // u.x = 0.5F * (u.x + x/u.x);
    // u.x = 0.5F * (u.x + x/u.x);
    u.x =       u.x + x/u.x;
    u.x = 0.25F*u.x + x/u.x;

    return u.x;
}

inline Float_t KFastMath::InvSqrt_Q3(const Float_t x, uint32_t n)
{
    const Float_t xhalf = 0.5F * x;

    union // get bits for floating value
    {
        Float_t x;
        Long_t i;
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

inline Float_t KFastMath::FastSqrt_Q3(const Float_t x, uint32_t n)
{
    return x * InvSqrt_Q3(x, n);
}

inline double KFastMath::FastSqrt_Q3(const double x, uint32_t n)
{
    return x * InvSqrt_Q3(x, n);
}

inline double KFastMath::FastExp(double y) {
    double d;
    *((int*)(&d) + 0) = 0;
    *((int*)(&d) + 1) = (int)(1512775 * y + 1072632447);
    return d;
}

inline double KFastMath::FastExp_Taylor(double x, int32_t n)
{
    if (n < 0) {
        return std::exp(x);
    }
    else {
        double result = 1;
        double pow_x;
        uint32_t i, j;
        for (i=1; i<=(uint32_t)n; ++i) {
            pow_x = x;
            for (j=1; j<i; ++j) {
                pow_x *= x;
            }
            result += pow_x / boost::math::factorial<double>(i);
        }
        return result;
    }
}


}


#endif /* KFASTMATH_H_ */
