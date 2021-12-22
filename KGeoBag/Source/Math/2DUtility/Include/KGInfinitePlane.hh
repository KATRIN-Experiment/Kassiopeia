#ifndef __KGInfinitePlane_H__
#define __KGInfinitePlane_H__

#include "KThreeVector.hh"
#include "KTwoVector.hh"

#define SMALLNUMBER 1e-9

namespace KGeoBag
{

/**
*
*@file KGInfinitePlane.hh
*@class KGInfinitePlane
*@brief  class to represent a plane with infinite extent,
* only to used as a helper in calculations, not useful as an actual object.
*@details
*
*<b>Revision History:<b>
*Date Name Brief Description
*Mon Jul 23 14:03:39 EDT 2012 J. Barrett (barrettj@mit.edu) First Version
*
*/


class KGInfinitePlane
{
  public:
    ///default constructor makes the untranslated/unrotated x-y plane
    KGInfinitePlane()
    {
        fOrigin = katrin::KThreeVector(0, 0, 0);
        fXAxis = katrin::KThreeVector(1, 0, 0);
        fYAxis = katrin::KThreeVector(0, 1, 0);
        fZAxis = katrin::KThreeVector(0, 0, 1);
    };

    ///construct a plane from a point and normal vector
    KGInfinitePlane(const katrin::KThreeVector& point, const katrin::KThreeVector& normal);

    //construct a plane from three points
    KGInfinitePlane(const katrin::KThreeVector& point0, const katrin::KThreeVector& point1,
                    const katrin::KThreeVector& point2);
    virtual ~KGInfinitePlane()
    {
        ;
    };

    //no initialization needed, all done in constructor
    virtual void Initialize()
    {
        ;
    };

    bool IsAbove(const katrin::KThreeVector& vec) const;

    virtual void NearestDistance(const katrin::KThreeVector& aPoint, double& aDistance) const;
    virtual void NearestPoint(const katrin::KThreeVector& aPoint, katrin::KThreeVector& aNearest) const;
    virtual void NearestNormal(const katrin::KThreeVector& /*aPoint*/, katrin::KThreeVector& aNormal) const;
    virtual void NearestIntersection(const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd,
                                     bool& aResult, katrin::KThreeVector& anIntersection) const;

    katrin::KTwoVector Project(const katrin::KThreeVector& aPoint) const;

    katrin::KThreeVector GetOrigin() const
    {
        return fOrigin;
    }
    katrin::KThreeVector GetXAxis() const
    {
        return fXAxis;
    };
    katrin::KThreeVector GetYAxis() const
    {
        return fYAxis;
    };
    katrin::KThreeVector GetZAxis() const
    {
        return fZAxis;
    };

    void SetOrigin(const katrin::KThreeVector& origin)
    {
        fOrigin = origin;
    }
    void SetXAxis(const katrin::KThreeVector& x_axis)
    {
        fXAxis = x_axis;
    };
    void SetYAxis(const katrin::KThreeVector& y_axis)
    {
        fYAxis = y_axis;
    };
    void SetZAxis(const katrin::KThreeVector& z_axis)
    {
        fZAxis = z_axis;
    };


    //static utility function for point-normal defined planes
    static double NearestDistance(const katrin::KThreeVector& origin, const katrin::KThreeVector& unit_normal,
                                  const katrin::KThreeVector& aPoint);

    static katrin::KThreeVector NearestPoint(const katrin::KThreeVector& origin,
                                              const katrin::KThreeVector& unit_normal,
                                              const katrin::KThreeVector& aPoint);

    static bool NearestIntersection(const katrin::KThreeVector& origin, const katrin::KThreeVector& unit_normal,
                                    const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd,
                                    katrin::KThreeVector& anIntersection, double& distance);


  protected:
    katrin::KThreeVector fOrigin;
    katrin::KThreeVector fXAxis;
    katrin::KThreeVector fYAxis;
    katrin::KThreeVector fZAxis;
};

}  // namespace KGeoBag


#endif /* __KGInfinitePlane_H__ */
