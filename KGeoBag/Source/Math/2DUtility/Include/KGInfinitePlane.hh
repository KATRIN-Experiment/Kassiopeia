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
        fOrigin = KGeoBag::KThreeVector(0, 0, 0);
        fXAxis = KGeoBag::KThreeVector(1, 0, 0);
        fYAxis = KGeoBag::KThreeVector(0, 1, 0);
        fZAxis = KGeoBag::KThreeVector(0, 0, 1);
    };

    ///construct a plane from a point and normal vector
    KGInfinitePlane(const KGeoBag::KThreeVector& point, const KGeoBag::KThreeVector& normal);

    //construct a plane from three points
    KGInfinitePlane(const KGeoBag::KThreeVector& point0, const KGeoBag::KThreeVector& point1,
                    const KGeoBag::KThreeVector& point2);
    virtual ~KGInfinitePlane()
    {
        ;
    };

    //no initialization needed, all done in constructor
    virtual void Initialize()
    {
        ;
    };

    bool IsAbove(const KGeoBag::KThreeVector& vec) const;

    virtual void NearestDistance(const KGeoBag::KThreeVector& aPoint, double& aDistance) const;
    virtual void NearestPoint(const KGeoBag::KThreeVector& aPoint, KGeoBag::KThreeVector& aNearest) const;
    virtual void NearestNormal(const KGeoBag::KThreeVector& /*aPoint*/, KGeoBag::KThreeVector& aNormal) const;
    virtual void NearestIntersection(const KGeoBag::KThreeVector& aStart, const KGeoBag::KThreeVector& anEnd,
                                     bool& aResult, KGeoBag::KThreeVector& anIntersection) const;

    KTwoVector Project(const KGeoBag::KThreeVector& aPoint) const;

    KGeoBag::KThreeVector GetOrigin() const
    {
        return fOrigin;
    }
    KGeoBag::KThreeVector GetXAxis() const
    {
        return fXAxis;
    };
    KGeoBag::KThreeVector GetYAxis() const
    {
        return fYAxis;
    };
    KGeoBag::KThreeVector GetZAxis() const
    {
        return fZAxis;
    };

    void SetOrigin(const KGeoBag::KThreeVector& origin)
    {
        fOrigin = origin;
    }
    void SetXAxis(const KGeoBag::KThreeVector& x_axis)
    {
        fXAxis = x_axis;
    };
    void SetYAxis(const KGeoBag::KThreeVector& y_axis)
    {
        fYAxis = y_axis;
    };
    void SetZAxis(const KGeoBag::KThreeVector& z_axis)
    {
        fZAxis = z_axis;
    };


    //static utility function for point-normal defined planes
    static double NearestDistance(const KGeoBag::KThreeVector& origin, const KGeoBag::KThreeVector& unit_normal,
                                  const KGeoBag::KThreeVector& aPoint);

    static KGeoBag::KThreeVector NearestPoint(const KGeoBag::KThreeVector& origin,
                                              const KGeoBag::KThreeVector& unit_normal,
                                              const KGeoBag::KThreeVector& aPoint);

    static bool NearestIntersection(const KGeoBag::KThreeVector& origin, const KGeoBag::KThreeVector& unit_normal,
                                    const KGeoBag::KThreeVector& aStart, const KGeoBag::KThreeVector& anEnd,
                                    KGeoBag::KThreeVector& anIntersection, double& distance);


  protected:
    KGeoBag::KThreeVector fOrigin;
    KGeoBag::KThreeVector fXAxis;
    KGeoBag::KThreeVector fYAxis;
    KGeoBag::KThreeVector fZAxis;
};

}  // namespace KGeoBag


#endif /* __KGInfinitePlane_H__ */
