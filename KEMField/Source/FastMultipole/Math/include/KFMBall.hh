#ifndef KFMBall_HH__
#define KFMBall_HH__

#include "KFMPoint.hh"

namespace KEMField
{

/*
*
*@file KFMBall.hh
*@class KFMBall
*@brief ball in euclidean space
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Aug 13 06:42:31 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<unsigned int NDIM> class KFMBall
{
  public:
    KFMBall()
    {
        for (unsigned int i = 0; i < NDIM + 1; i++) {
            fData[i] = 0.0;
        }
    };

    KFMBall(const KFMBall& copyObject)
    {
        for (unsigned int i = 0; i < NDIM + 1; i++) {
            fData[i] = copyObject.fData[i];
        }
    }


    KFMBall(KFMPoint<NDIM> center, double radius)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fData[i] = center[i];
        }
        fData[NDIM] = radius;
    }

    virtual ~KFMBall(){};

    unsigned int GetDimension() const
    {
        return NDIM;
    };

    double GetRadius() const
    {
        return fData[NDIM];
    };
    void SetRadius(double r)
    {
        fData[NDIM] = r;
    };

    KFMPoint<NDIM> GetCenter() const
    {
        return KFMPoint<NDIM>(fData);
    };
    void SetCenter(KFMPoint<NDIM> center)
    {
        for (unsigned int i = 0; i < NDIM; i++) {
            fData[i] = center[i];
        }
    };

    //navigation
    bool PointIsInside(const double* p) const
    {

        KFMPoint<NDIM> del = KFMPoint<NDIM>(p) - KFMPoint<NDIM>(fData);
        double del_mag = del.Magnitude();

        if (del_mag < fData[NDIM]) {
            return true;
        }
        else {
            return false;
        }
    }

    bool BallIsInside(const KFMBall<NDIM>* ball) const
    {
        KFMPoint<NDIM> del = ball->GetCenter() - KFMPoint<NDIM>(fData);
        double del_mag = del.Magnitude();
        double br = ball->GetRadius();

        if (del_mag + br < fData[NDIM]) {
            return true;
        }
        else {
            return false;
        }
    }

    bool BallIsOutside(const KFMBall<NDIM>* ball) const
    {
        KFMPoint<NDIM> del = ball->GetCenter() - KFMPoint<NDIM>(fData);
        double del_mag = del.Magnitude();

        if (del_mag > (fData[NDIM] + ball->GetRadius())) {
            return true;
        }
        else {
            return false;
        }
    }


    //access elements
    double& operator[](unsigned int i);
    const double& operator[](unsigned int i) const;


    inline KFMBall& operator=(const KFMBall& rhs)
    {
        if (&rhs != this) {
            for (unsigned int i = 0; i < NDIM + 1; i++) {
                fData[i] = rhs.fData[i];
            }
        }
        return *this;
    }


  private:
    double fData[NDIM + 1];  //center position + length, last element is length
};


template<unsigned int NDIM> inline double& KFMBall<NDIM>::operator[](unsigned int i)
{
    return fData[i];
}

template<unsigned int NDIM> inline const double& KFMBall<NDIM>::operator[](unsigned int i) const
{
    return fData[i];
}


}  // namespace KEMField


#endif /* KFMBall_H__ */
