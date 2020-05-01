#ifndef __KGInfinitePlane_H__
#define __KGInfinitePlane_H__

#include "KTransformation.hh"

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
        fOrigin = KThreeVector(0, 0, 0);
        fXAxis = KThreeVector(1, 0, 0);
        fYAxis = KThreeVector(0, 1, 0);
        fZAxis = KThreeVector(0, 0, 1);
    };

    ///construct a plane from a point and normal vector
    KGInfinitePlane(const KThreeVector& point, const KThreeVector& normal);

    //construct a plane from three points
    KGInfinitePlane(const KThreeVector& point0, const KThreeVector& point1, const KThreeVector& point2);
    virtual ~KGInfinitePlane()
    {
        ;
    };

    //no initialization needed, all done in constructor
    virtual void Initialize()
    {
        ;
    };

    bool IsAbove(const KThreeVector vec) const;

    virtual void NearestDistance(const KThreeVector& aPoint, double& aDistance) const;
    virtual void NearestPoint(const KThreeVector& aPoint, KThreeVector& aNearest) const;
    virtual void NearestNormal(const KThreeVector& /*aPoint*/, KThreeVector& aNormal) const;
    virtual void NearestIntersection(const KThreeVector& aStart, const KThreeVector& anEnd, bool& aResult,
                                     KThreeVector& anIntersection) const;

    KTwoVector Project(const KThreeVector& aPoint) const;

    KThreeVector GetOrigin() const
    {
        return fOrigin;
    }
    KThreeVector GetXAxis() const
    {
        return fXAxis;
    };
    KThreeVector GetYAxis() const
    {
        return fYAxis;
    };
    KThreeVector GetZAxis() const
    {
        return fZAxis;
    };

    void SetOrigin(const KThreeVector& origin)
    {
        fOrigin = origin;
    }
    void SetXAxis(const KThreeVector x_axis)
    {
        fXAxis = x_axis;
    };
    void SetYAxis(const KThreeVector y_axis)
    {
        fYAxis = y_axis;
    };
    void SetZAxis(const KThreeVector z_axis)
    {
        fZAxis = z_axis;
    };


    //static utility function for point-normal defined planes
    static double NearestDistance(const KThreeVector& origin, const KThreeVector& unit_normal,
                                  const KThreeVector& aPoint);

    static KThreeVector NearestPoint(const KThreeVector& origin, const KThreeVector& unit_normal,
                                     const KThreeVector& aPoint);

    static bool NearestIntersection(const KThreeVector& origin, const KThreeVector& unit_normal,
                                    const KThreeVector& aStart, const KThreeVector& anEnd, KThreeVector& anIntersection,
                                    double& distance);


  protected:
    KThreeVector fOrigin;
    KThreeVector fXAxis;
    KThreeVector fYAxis;
    KThreeVector fZAxis;
};

}  // namespace KGeoBag


#endif /* __KGInfinitePlane_H__ */
