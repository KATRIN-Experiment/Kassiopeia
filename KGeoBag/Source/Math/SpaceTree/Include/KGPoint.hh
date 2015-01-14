#ifndef KGPoint_HH__
#define KGPoint_HH__

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

template<size_t NDIM>
class KGPoint
{
    public:

        KGPoint()
        {
            for(size_t i=0; i<NDIM; i++)
            {
                fData[i] = 0; //init to zero
            }
        };

        KGPoint(const double* p)
        {
            for(size_t i=0; i<NDIM; i++)
            {
                fData[i] = p[i];
            }
        }

        KGPoint( const KGPoint<NDIM>& p )
        {
            for(size_t i=0; i<NDIM; i++)
            {
                fData[i] = p[i];
            }
        }

        virtual ~KGPoint(){};

        size_t GetDimension() const {return NDIM;};

        //cast to double array
        operator double* ();
        operator const double* () const;

        //access elements
        double& operator[](size_t i);
        const double& operator[](size_t i) const;

        //assignment
        inline KGPoint<NDIM>& operator=(const KGPoint<NDIM>& p)
        {
            for(size_t i=0; i<NDIM; i++)
            {
                fData[ i ] = p.fData[ i ];
            }
            return *this;
        }


        double MagnitudeSquared() const
        {
            double val = 0;
            for(size_t i=0; i<NDIM; i++)
            {
                val += fData[i]*fData[i];
            }
            return val;
        }

        double Magnitude() const
        {
            double val = 0;
            for(size_t i=0; i<NDIM; i++)
            {
                val += fData[i]*fData[i];
            }
            return std::sqrt(val);
        }


    private:

        double fData[NDIM];

};


template<size_t NDIM>
inline KGPoint<NDIM>::operator double* ()
{
    return fData;
}

template<size_t NDIM>
inline KGPoint<NDIM>::operator const double* () const
{
    return fData;
}

template<size_t NDIM>
inline double& KGPoint<NDIM>::operator[](size_t i)
{
    return fData[i];
}

template<size_t NDIM>
inline const double& KGPoint<NDIM>::operator[](size_t i) const
{
    return fData[i];
}

template<size_t NDIM>
inline KGPoint<NDIM> operator+( const KGPoint<NDIM>& aLeft, const KGPoint<NDIM>& aRight )
{
    KGPoint<NDIM> aResult( aLeft );
    for(size_t i=0; i<NDIM; i++)
    {
        aResult[i] += aRight[i];
    }
    return aResult;
}

template<size_t NDIM>
inline KGPoint<NDIM>& operator+=( KGPoint<NDIM>& aLeft, const KGPoint<NDIM>& aRight )
{
    for(size_t i=0; i<NDIM; i++)
    {
        aLeft[i] += aRight[i];
    }
    return aLeft;
}

template<size_t NDIM>
inline KGPoint<NDIM>  operator-( const KGPoint<NDIM>& aLeft, const KGPoint<NDIM>& aRight )
{
    KGPoint<NDIM> aResult( aLeft );
    for(size_t i=0; i<NDIM; i++)
    {
        aResult[i] -= aRight[i];
    }
    return aResult;
}

template<size_t NDIM>
inline KGPoint<NDIM>& operator-=( KGPoint<NDIM>& aLeft, const KGPoint<NDIM>& aRight )
{
    for(size_t i=0; i<NDIM; i++)
    {
        aLeft[i] -= aRight[i];
    }
    return aLeft;
}

template<size_t NDIM>
inline double operator*( const KGPoint<NDIM>& aLeft, const KGPoint<NDIM>& aRight )
{
    double val = 0;
    for(size_t i=0; i<NDIM; i++)
    {
        val += aLeft[i] * aRight[i];
    }
    return val;
}

template<size_t NDIM>
inline KGPoint<NDIM>  operator*( register double aScalar, const KGPoint<NDIM>& aVector )
{
    KGPoint<NDIM>  aResult( aVector );
    for(size_t i=0; i<NDIM; i++)
    {
        aResult[i] *= aScalar;
    }
    return aResult;
}

template<size_t NDIM>
inline KGPoint<NDIM> operator*( const KGPoint<NDIM>& aVector, register double aScalar )
{
    KGPoint<NDIM>  aResult( aVector );
    for(size_t i=0; i<NDIM; i++)
    {
        aResult[i] *= aScalar;
    }
    return aResult;
}

template<size_t NDIM>
inline KGPoint<NDIM>& operator*=( KGPoint<NDIM>& aVector, register double aScalar )
{
    for(size_t i=0; i<NDIM; i++)
    {
        aVector[i] *= aScalar;
    }
    return aVector;
}

template<size_t NDIM>
inline KGPoint<NDIM> operator/( const KGPoint<NDIM>& aVector, register double aScalar )
{
    KGPoint<NDIM>  aResult( aVector );
    for(size_t i=0; i<NDIM; i++)
    {
        aResult[i] /= aScalar;
    }
    return aResult;
}

template<size_t NDIM>
inline KGPoint<NDIM>& operator/=( KGPoint<NDIM>& aVector, register double aScalar )
{
    for(size_t i=0; i<NDIM; i++)
    {
        aVector[i] /= aScalar;
    }
    return aVector;
}


}//end of KGeoBag

#endif /* KGPoint_H__ */
