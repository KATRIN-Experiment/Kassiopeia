#include "KGInfinitePlane.hh"

#include <iostream>

namespace KGeoBag
{


KGInfinitePlane::KGInfinitePlane(const KThreeVector& point, const KThreeVector& normal)
{
    fOrigin = point;
    fZAxis = normal.Unit();

    //since orientation is not unique this is completely arbitrary here
    fXAxis = KThreeVector(1,0,0);

    if( (fXAxis - fZAxis).Magnitude() < 1e-6 ) //if they are ~equal add an orthogonal component
    {
        fXAxis += KThreeVector(0,1,0);
    }

    //now do gram-schmidt

    fXAxis = (fXAxis - (fXAxis.Dot(fZAxis))*fZAxis ).Unit();

    //cross prod
    fYAxis = (fZAxis.Cross(fXAxis)).Unit();

//    std::cout<<fOrigin<<std::endl;
//    std::cout<<fZAxis<<std::endl;


}

KGInfinitePlane::KGInfinitePlane(const KThreeVector& point0, const KThreeVector& point1, const KThreeVector& point2)
{
    KThreeVector normal = ( ( (point1 - point0).Unit() ).Cross( (point2 - point0).Unit() ) ).Unit();
    //first we set the local origin to the point which defines the plane
    fOrigin = point0;
    //align z-axis with normal
    fZAxis = normal;
    //align the x-axis to point from point0 to point1;
    fXAxis = (point1 - point0).Unit();
    //compute y axis
    fYAxis = (fZAxis.Cross(fXAxis)).Unit();
}

void
KGInfinitePlane::NearestDistance( const KThreeVector& aPoint, double& aDistance ) const
{
    aDistance = fabs( fZAxis.Dot( aPoint - fOrigin ) );
}

void
KGInfinitePlane::NearestPoint( const KThreeVector& aPoint, KThreeVector& aNearest ) const
{
    double dist = fZAxis.Dot( aPoint - fOrigin );
    aNearest = aPoint + dist*fZAxis;
}

void
KGInfinitePlane::NearestNormal( const KThreeVector& aPoint, KThreeVector& aNormal ) const
{
    if( fZAxis.Dot( aPoint - fOrigin ) > 0 )
    {
        aNormal = fZAxis; //we are above the plane
    }
    else
    {
        aNormal = -1*fZAxis; //we are below the plane
    }
}

void
KGInfinitePlane::NearestIntersection( const KThreeVector& aStart, const KThreeVector& anEnd, bool& aResult, KThreeVector& anIntersection ) const
{
    KThreeVector v = (anEnd - aStart).Unit();
    double len = v.Magnitude();
    double ndotv = fZAxis.Dot(v);

    aResult = false;

    if( fabs(ndotv) < SMALLNUMBER)
    {
        //The line segment is ~parallel to the plane, note the line segment
        //could be in the plane, but this results in a infinite number of
        //intersections we do not check for this, since with floating point
        //math this is pretty unlikely.
        return;
    }

    double t = ( fZAxis.Dot(fOrigin) - fZAxis.Dot(aStart) )/ndotv;

    if( t < 0)
    {
        //plane is behind the first point of the line segment
        //no intersection
        return;
    }

    if( t > len )
    {
        //intersection is beyond the last point of the line
        //segment, so no intersection
        return;
    }

    //passed all the checks, then there must be an intersection given by;
    aResult = true;
    anIntersection = aStart + t*v;

}

bool
KGInfinitePlane::IsAbove(const KThreeVector vec) const
{
    if((vec - fOrigin).Dot(fZAxis)> 0){return true;};
    return false;
}


KTwoVector
KGInfinitePlane::Project(const KThreeVector& aPoint) const
{
    KThreeVector del = aPoint - fOrigin;
    return KTwoVector(del.Dot(fXAxis), del.Dot(fYAxis));
}


}
