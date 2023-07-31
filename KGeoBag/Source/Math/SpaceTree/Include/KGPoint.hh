#ifndef KGPoint_HH__
#define KGPoint_HH__

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

namespace KGeoBag
{

/*
*
*@file KGPoint.hh
*@class KGPoint
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Aug 13 05:40:50 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<size_t NDIM = 3> class KGPoint
{
  public:
    KGPoint()
    {
        fData.fill(0.0);
    }

    KGPoint(const double* p)
    {
        for (unsigned int i=0; i<NDIM; i++) {
            fData[i] = p[i];
        }
    }

    virtual ~KGPoint() = default;

    size_t GetDimension() const
    {
        return NDIM;
    }

    // cast to double array
    operator double*()
    {
        return fData.data();
    }
    operator const double*() const
    {
        return fData.data();
    }

    double MagnitudeSquared() const
    {
        double result = 0.0;
        for (size_t i = 0; i < NDIM; i++)
            result += fData[i] * fData[i];
        return result;
    }

    double Magnitude() const
    {
        return std::sqrt(MagnitudeSquared());
    }

  private:
    std::array<double, NDIM> fData;
};


template<size_t NDIM> inline KGPoint<NDIM> operator+(const KGPoint<NDIM>& aLeft, const KGPoint<NDIM>& aRight)
{
    KGPoint<NDIM> aResult(aLeft);
    for (size_t i = 0; i < NDIM; i++)
        aResult[i] += aRight[i];
    return aResult;
}

template<size_t NDIM> inline KGPoint<NDIM>& operator+=(KGPoint<NDIM>& aLeft, const KGPoint<NDIM>& aRight)
{
    for (size_t i = 0; i < NDIM; i++)
        aLeft[i] += aRight[i];
    return aLeft;
}

template<size_t NDIM> inline KGPoint<NDIM> operator-(const KGPoint<NDIM>& aLeft, const KGPoint<NDIM>& aRight)
{
    KGPoint<NDIM> aResult(aLeft);
    for (size_t i = 0; i < NDIM; i++)
        aResult[i] -= aRight[i];
    return aResult;
}

template<size_t NDIM> inline KGPoint<NDIM>& operator-=(KGPoint<NDIM>& aLeft, const KGPoint<NDIM>& aRight)
{
    for (size_t i = 0; i < NDIM; i++)
        aLeft[i] -= aRight[i];
    return aLeft;
}

template<size_t NDIM> inline double operator*(const KGPoint<NDIM>& aLeft, const KGPoint<NDIM>& aRight)
{
    double val = 0;
    for (size_t i = 0; i < NDIM; i++)
        val += aLeft[i] * aRight[i];
    return val;
}

template<size_t NDIM> inline KGPoint<NDIM> operator*(double aScalar, const KGPoint<NDIM>& aVector)
{
    KGPoint<NDIM> aResult(aVector);
    for (size_t i = 0; i < NDIM; i++)
        aResult[i] *= aScalar;
    return aResult;
}

template<size_t NDIM> inline KGPoint<NDIM> operator*(const KGPoint<NDIM>& aVector, double aScalar)
{
    KGPoint<NDIM> aResult(aVector);
    for (size_t i = 0; i < NDIM; i++)
        aResult[i] *= aScalar;
    return aResult;
}

template<size_t NDIM> inline KGPoint<NDIM>& operator*=(KGPoint<NDIM>& aVector, double aScalar)
{
    for (size_t i = 0; i < NDIM; i++)
        aVector[i] *= aScalar;
    return aVector;
}

template<size_t NDIM> inline KGPoint<NDIM> operator/(const KGPoint<NDIM>& aVector, double aScalar)
{
    KGPoint<NDIM> aResult(aVector);
    for (size_t i = 0; i < NDIM; i++)
        aResult[i] /= aScalar;
    return aResult;
}

template<size_t NDIM> inline KGPoint<NDIM>& operator/=(KGPoint<NDIM>& aVector, double aScalar)
{
    for (size_t i = 0; i < NDIM; i++)
        aVector[i] /= aScalar;
    return aVector;
}


}  // namespace KGeoBag

#endif /* KGPoint_H__ */
