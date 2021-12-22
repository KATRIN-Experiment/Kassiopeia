#include "KGInfinitePlane.hh"

#include <iostream>

using katrin::KTwoVector;

namespace KGeoBag
{


KGInfinitePlane::KGInfinitePlane(const katrin::KThreeVector& point, const katrin::KThreeVector& normal)
{
    fOrigin = point;
    fZAxis = normal.Unit();

    //since orientation is not unique this is completely arbitrary here
    fXAxis = katrin::KThreeVector(1, 0, 0);

    if ((fXAxis - fZAxis).Magnitude() < 1e-6)  //if they are ~equal add an orthogonal component
    {
        fXAxis += katrin::KThreeVector(0, 1, 0);
    }

    //now do gram-schmidt

    fXAxis = (fXAxis - (fXAxis.Dot(fZAxis)) * fZAxis).Unit();

    //cross prod
    fYAxis = (fZAxis.Cross(fXAxis)).Unit();

    //    std::cout<<fOrigin<<std::endl;
    //    std::cout<<fZAxis<<std::endl;
}

KGInfinitePlane::KGInfinitePlane(const katrin::KThreeVector& point0, const katrin::KThreeVector& point1, const katrin::KThreeVector& point2)
{
    katrin::KThreeVector normal = (((point1 - point0).Unit()).Cross((point2 - point0).Unit())).Unit();
    //first we set the local origin to the point which defines the plane
    fOrigin = point0;
    //align z-axis with normal
    fZAxis = normal;
    //align the x-axis to point from point0 to point1;
    fXAxis = (point1 - point0).Unit();
    //compute y axis
    fYAxis = (fZAxis.Cross(fXAxis)).Unit();
}

void KGInfinitePlane::NearestDistance(const katrin::KThreeVector& aPoint, double& aDistance) const
{
    aDistance = fabs(fZAxis.Dot(aPoint - fOrigin));
}

void KGInfinitePlane::NearestPoint(const katrin::KThreeVector& aPoint, katrin::KThreeVector& aNearest) const
{
    double dist = fZAxis.Dot(aPoint - fOrigin);
    aNearest = aPoint + dist * fZAxis;
}

void KGInfinitePlane::NearestNormal(const katrin::KThreeVector& /*aPoint*/, katrin::KThreeVector& aNormal) const
{
    aNormal = fZAxis;
}

void KGInfinitePlane::NearestIntersection(const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd, bool& aResult,
                                          katrin::KThreeVector& anIntersection) const
{
    katrin::KThreeVector v = (anEnd - aStart).Unit();
    double len = v.Magnitude();
    double ndotv = fZAxis.Dot(v);

    aResult = false;

    if (fabs(ndotv) < SMALLNUMBER) {
        //The line segment is ~parallel to the plane, note the line segment
        //could be in the plane, but this results in a infinite number of
        //intersections we do not check for this, since with floating point
        //math this is pretty unlikely.
        return;
    }

    double t = (fZAxis.Dot(fOrigin) - fZAxis.Dot(aStart)) / ndotv;

    if (t < 0) {
        //plane is behind the first point of the line segment
        //no intersection
        return;
    }

    if (t > len) {
        //intersection is beyond the last point of the line
        //segment, so no intersection
        return;
    }

    //passed all the checks, then there must be an intersection given by;
    aResult = true;
    anIntersection = aStart + t * v;
}

bool KGInfinitePlane::IsAbove(const katrin::KThreeVector& vec) const
{
    if ((vec - fOrigin).Dot(fZAxis) > 0) {
        return true;
    };
    return false;
}


KTwoVector KGInfinitePlane::Project(const katrin::KThreeVector& aPoint) const
{
    katrin::KThreeVector del = aPoint - fOrigin;
    return KTwoVector(del.Dot(fXAxis), del.Dot(fYAxis));
}


//static utility function for point-normal defined planes
double KGInfinitePlane::NearestDistance(const katrin::KThreeVector& origin, const katrin::KThreeVector& unit_normal,
                                        const katrin::KThreeVector& aPoint)
{
    return fabs(unit_normal.Dot(aPoint - origin));
}

katrin::KThreeVector KGInfinitePlane::NearestPoint(const katrin::KThreeVector& origin, const katrin::KThreeVector& unit_normal,
                                           const katrin::KThreeVector& aPoint)
{
    double signed_dist = unit_normal.Dot(aPoint - origin);
    return aPoint + signed_dist * unit_normal;
}

bool KGInfinitePlane::NearestIntersection(const katrin::KThreeVector& origin, const katrin::KThreeVector& unit_normal,
                                          const katrin::KThreeVector& aStart, const katrin::KThreeVector& anEnd,
                                          katrin::KThreeVector& anIntersection, double& distance)
{
    katrin::KThreeVector v = (anEnd - aStart).Unit();
    double len = v.Magnitude();
    double ndotv = unit_normal.Dot(v);

    if (fabs(ndotv) < SMALLNUMBER) {
        //The line segment is ~parallel to the plane, note the line segment
        //could be in the plane, but this results in a infinite number of
        //intersections we do not check for this, since with floating point
        //math this is pretty unlikely.
        return false;
    }

    double t = (unit_normal.Dot(origin) - unit_normal.Dot(aStart)) / ndotv;

    //plane is behind the first point of the line segment
    //no intersection
    if (t < 0) {
        return false;
    };

    //intersection is beyond the last point of the line
    //segment, so no intersection
    if (t > len) {
        return false;
    }

    //passed all the checks, then there must be an intersection given by;
    anIntersection = aStart + t * v;
    distance = t;
    return true;
}


}  // namespace KGeoBag
