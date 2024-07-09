#ifndef KGBall_HH__
#define KGBall_HH__

#include "KGPoint.hh"

namespace KGeoBag
{

/*
*
*@file KGBall.hh
*@class KGBall
*@brief ball in euclidean space
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Tue Aug 13 06:42:31 EDT 2013 J. Barrett (barrettj@mit.edu) First Version
*
*/

template<size_t NDIM> class KGBall
{
  public:
    KGBall()
    {
        fData.fill(0.0);
    };


    KGBall(KGPoint<NDIM> center, double radius)
    {
        for (size_t i = 0; i < NDIM; i++) {
            fData[i] = center[i];
        }
        fData[NDIM] = radius;
    }

    virtual ~KGBall() = default;

    size_t GetDimension() const
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

    KGPoint<NDIM> GetCenter() const
    {
        return KGPoint<NDIM>(fData.data());
    };
    void SetCenter(const KGPoint<NDIM>& center)
    {
        for (size_t i = 0; i < NDIM; i++) {
            fData[i] = center[i];
        }
    };

    //navigation
    bool PointIsInside(const double* p) const
    {
        double del;
        double dist2 = 0.0;
        for (unsigned int i = 0; i < NDIM; i++) {
            del = p[i] - fData[i];
            dist2 += del * del;
        }
        if (dist2 < fData[NDIM] * fData[NDIM]) {
            return true;
        }
        return false;

        // KGPoint<NDIM> del = KGPoint<NDIM>(p) - KGPoint<NDIM>(fData);
        // double distance2 = del*del;
        //
        // double distance = std::sqrt(distance2); //have to perform the sqrt b/c floating point math is picky
        //
        // if(distance - fData[NDIM] > 0.0)
        // {
        //     return false;
        // }
        // else
        // {
        //     return true;
        // }
    }

    bool BallIsInside(const KGBall<NDIM>* ball) const
    {
        KGPoint<NDIM> del = ball->GetCenter() - KGPoint<NDIM>(fData);
        double del_mag = del.Magnitude();
        double br = ball->GetRadius();
        double distance2 = (del_mag + br) * (del_mag + br);

        double distance = std::sqrt(distance2);  //have to perform the sqrt b/c floating point math is picky

        if (distance - fData[NDIM] > 0.0) {
            return false;
        }
        else {
            return true;
        }
    }


    //access elements
    double& operator[](size_t i);
    const double& operator[](size_t i) const;

  private:
    std::array<double, NDIM + 1> fData;  //center position + length, last element is length
};


template<size_t NDIM> inline double& KGBall<NDIM>::operator[](size_t i)
{
    return fData[i];
}

template<size_t NDIM> inline const double& KGBall<NDIM>::operator[](size_t i) const
{
    return fData[i];
}


}  // namespace KGeoBag


#endif /* KGBall_H__ */
