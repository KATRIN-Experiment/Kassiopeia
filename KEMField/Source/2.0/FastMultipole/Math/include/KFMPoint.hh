#ifndef KFMPoint_HH__
#define KFMPoint_HH__

#include <cmath>
#include <cstddef>


namespace KEMField
{

/*
*
*@file KFMPoint.hh
*@class KFMPoint
*@brief
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Aug 13 05:40:50 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM>
class KFMPoint
{
    public:

        KFMPoint()
        {
            for(unsigned int i=0; i<NDIM; i++)
            {
                fData[i] = 0; //init to zero
            }
        };

        KFMPoint(const double* p)
        {
            for(unsigned int i=0; i<NDIM; i++)
            {
                fData[i] = p[i];
            }
        }

        KFMPoint( const KFMPoint<NDIM>& p )
        {
            for(unsigned int i=0; i<NDIM; i++)
            {
                fData[i] = p[i];
            }
        }

        virtual ~KFMPoint(){};

        unsigned int GetDimension() const {return NDIM;};

        //cast to double array
        operator double* ();
        operator const double* () const;

        //assignment
        inline KFMPoint<NDIM>& operator=(const KFMPoint<NDIM>& p)
        {
            for(unsigned int i=0; i<NDIM; i++)
            {
                fData[ i ] = p.fData[ i ];
            }
            return *this;
        }


        double MagnitudeSquared() const
        {
            double val = 0;
            for(unsigned int i=0; i<NDIM; i++)
            {
                val += fData[i]*fData[i];
            }
            return val;
        }

        double Magnitude() const
        {
            double val = 0;
            for(unsigned int i=0; i<NDIM; i++)
            {
                val += fData[i]*fData[i];
            }
            return std::sqrt(val);
        }

        double Dot(const KFMPoint<NDIM>& p) const
        {
            double val = 0;
            for(unsigned int i=0; i<NDIM; i++)
            {
                val += fData[i]*p[i];
            }
            return val;
        }

        KFMPoint<NDIM> Unit() const
        {
            KFMPoint<NDIM> val;
            double mag = this->Magnitude();
            for(unsigned int i=0; i<NDIM; i++)
            {
                val[i] = fData[i]/mag;
            }
            return val;
        }



    private:

        double fData[NDIM];

};


template<unsigned int NDIM>
inline KFMPoint<NDIM>::operator double* ()
{
    return fData;
}

template<unsigned int NDIM>
inline KFMPoint<NDIM>::operator const double* () const
{
    return fData;
}

//template<unsigned int NDIM>
//inline double& KFMPoint<NDIM>::operator[](unsigned int i)
//{
//    return fData[i];
//}

//template<unsigned int NDIM>
//inline const double& KFMPoint<NDIM>::operator[](unsigned int i) const
//{
//    return fData[i];
//}

template<unsigned int NDIM>
inline KFMPoint<NDIM> operator+( const KFMPoint<NDIM>& aLeft, const KFMPoint<NDIM>& aRight )
{
    KFMPoint<NDIM> aResult( aLeft );
    for(unsigned int i=0; i<NDIM; i++)
    {
        aResult[i] += aRight[i];
    }
    return aResult;
}

template<unsigned int NDIM>
inline KFMPoint<NDIM>& operator+=( KFMPoint<NDIM>& aLeft, const KFMPoint<NDIM>& aRight )
{
    for(unsigned int i=0; i<NDIM; i++)
    {
        aLeft[i] += aRight[i];
    }
    return aLeft;
}

template<unsigned int NDIM>
inline KFMPoint<NDIM>  operator-( const KFMPoint<NDIM>& aLeft, const KFMPoint<NDIM>& aRight )
{
    KFMPoint<NDIM> aResult( aLeft );
    for(unsigned int i=0; i<NDIM; i++)
    {
        aResult[i] -= aRight[i];
    }
    return aResult;
}

template<unsigned int NDIM>
inline KFMPoint<NDIM>& operator-=( KFMPoint<NDIM>& aLeft, const KFMPoint<NDIM>& aRight )
{
    for(unsigned int i=0; i<NDIM; i++)
    {
        aLeft[i] -= aRight[i];
    }
    return aLeft;
}

template<unsigned int NDIM>
inline double operator*( const KFMPoint<NDIM>& aLeft, const KFMPoint<NDIM>& aRight )
{
    double val = 0;
    for(unsigned int i=0; i<NDIM; i++)
    {
        val += aLeft[i] * aRight[i];
    }
    return val;
}

template<unsigned int NDIM>
inline KFMPoint<NDIM>  operator*( register double aScalar, const KFMPoint<NDIM>& aVector )
{
    KFMPoint<NDIM>  aResult( aVector );
    for(unsigned int i=0; i<NDIM; i++)
    {
        aResult[i] *= aScalar;
    }
    return aResult;
}

template<unsigned int NDIM>
inline KFMPoint<NDIM> operator*( const KFMPoint<NDIM>& aVector, register double aScalar )
{
    KFMPoint<NDIM>  aResult( aVector );
    for(unsigned int i=0; i<NDIM; i++)
    {
        aResult[i] *= aScalar;
    }
    return aResult;
}

template<unsigned int NDIM>
inline KFMPoint<NDIM>& operator*=( KFMPoint<NDIM>& aVector, register double aScalar )
{
    for(unsigned int i=0; i<NDIM; i++)
    {
        aVector[i] *= aScalar;
    }
    return aVector;
}

template<unsigned int NDIM>
inline KFMPoint<NDIM> operator/( const KFMPoint<NDIM>& aVector, register double aScalar )
{
    KFMPoint<NDIM>  aResult( aVector );
    for(unsigned int i=0; i<NDIM; i++)
    {
        aResult[i] /= aScalar;
    }
    return aResult;
}

template<unsigned int NDIM>
inline KFMPoint<NDIM>& operator/=( KFMPoint<NDIM>& aVector, register double aScalar )
{
    for(unsigned int i=0; i<NDIM; i++)
    {
        aVector[i] /= aScalar;
    }
    return aVector;
}


}//end of KEMField

#endif /* KFMPoint_H__ */
