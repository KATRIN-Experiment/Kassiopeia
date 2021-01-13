#ifndef __KG2DShape_H__
#define __KG2DShape_H__

#include "KTwoMatrix.hh"
#include "KTwoVector.hh"

#include <iostream>

namespace KGeoBag
{

/**
*
*@file KG2DShape.hh
*@class KG2DShape
*@brief 2-dimensional rip-off of KGAbstractShape with no transformations allowed
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Sat Jul 28 20:47:47 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KG2DShape
{
  public:
    KG2DShape()
    {
        ;
    };
    virtual ~KG2DShape()
    {
        ;
    };

    virtual void Initialize()
    {
        ;
    };

    //**************
    //visitor system
    //**************

  public:
    //****************
    //geometric system
    //****************

  public:
    virtual void NearestDistance(const KTwoVector& aPoint, double& aDistance) const = 0;
    virtual KTwoVector Point(const KTwoVector& aPoint) const = 0;
    virtual KTwoVector Normal(const KTwoVector& aPoint) const = 0;
    virtual void NearestIntersection(const KTwoVector& aStart, const KTwoVector& anEnd, bool& aResult,
                                     KTwoVector& anIntersection) const = 0;


    //these utilities probably do not belong here
    //here they are anyways
    static double Limit_to_0_to_2pi(double angle)
    {
        double val = angle;
        if (val < 0) {
            int n = std::fabs(val / 2.0 * (katrin::KConst::Pi()));
            val += n * (2.0 * katrin::KConst::Pi());
            return val;
        }

        if (val > 2.0 * katrin::KConst::Pi()) {
            int n = std::fabs(val / 2.0 * (katrin::KConst::Pi()));
            val -= n * (2.0 * katrin::KConst::Pi());
            return val;
        }

        return val;
    }

    static KTwoVector Rotate_vector_by_angle(const KTwoVector& vec, double angle)
    {
        //expects angles in radian
        double c = std::cos(angle);
        double s = std::sin(angle);
        KTwoMatrix mx(c, -1 * s, s, c);
        return mx * vec;
    }
};

}  // namespace KGeoBag

#endif /* __KG2DShape_H__ */
