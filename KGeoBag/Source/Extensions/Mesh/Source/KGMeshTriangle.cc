#include "KGMeshTriangle.hh"

#include "KTwoVector.hh"

#include <vector>
#include <math.h>

#define TRIANGLE_EPS 1e-6

namespace KGeoBag
{
    KGMeshTriangle::KGMeshTriangle( const double& fA, const double& fB, const KThreeVector& p0, const KThreeVector& n1, const KThreeVector& n2 ) :
            KGMeshElement(),
            fA( fA ),
            fB( fB ),
            fP0( p0 ),
            fN1( n1 ),
            fN2( n2 )
    {
    }
    KGMeshTriangle::KGMeshTriangle( const KThreeVector& p0, const KThreeVector& p1, const KThreeVector& p2 ) :
            KGMeshElement()
    {
        fP0 = p0;
        fN1 = p1 - p0;
        fA = fN1.Magnitude();
        fN1 = fN1.Unit();
        fN2 = p2 - p0;
        fB = fN2.Magnitude();
        fN2 = fN2.Unit();
    }
    KGMeshTriangle::~KGMeshTriangle()
    {
    }

    double KGMeshTriangle::Area() const
    {
        return .5 * fA * fB * fN1.Cross( fN2 ).Magnitude();
    }
    double KGMeshTriangle::Aspect() const
    {

        double c, max;
        double delx, dely, delz;
        double p1[ 3 ];
        double p2[ 3 ];

        for( int i = 0; i < 3; i++ )
        {
            p1[ i ] = fP0[ i ] + fA * fN1[ i ];
            p2[ i ] = fP0[ i ] + fB * fN2[ i ];
        }

        delx = p1[ 0 ] - p2[ 0 ];
        dely = p1[ 1 ] - p2[ 1 ];
        delz = p1[ 2 ] - p2[ 2 ];
        c = std::sqrt( delx * delx + dely * dely + delz * delz );

        KThreeVector PA;
        KThreeVector PB;
        KThreeVector PC;
        KThreeVector V;
        KThreeVector X;
        KThreeVector Y;
        KThreeVector Q;
        KThreeVector SUB;

        //find the longest size:
        if( fA > fB )
        {
            max = fA;
            PA = KThreeVector( p2[ 0 ], p2[ 1 ], p2[ 2 ] );
            PB = KThreeVector( fP0[ 0 ], fP0[ 1 ], fP0[ 2 ] );
            PC = KThreeVector( p1[ 0 ], p1[ 1 ], p1[ 2 ] );
        }
        else
        {
            max = fB;
            PA = KThreeVector( p1[ 0 ], p1[ 1 ], p1[ 2 ] );
            PB = KThreeVector( p2[ 0 ], p2[ 1 ], p2[ 2 ] );
            PC = KThreeVector( fP0[ 0 ], fP0[ 1 ], fP0[ 2 ] );
        }

        if( c > max )
        {
            max = c;
            PA = KThreeVector( fP0[ 0 ], fP0[ 1 ], fP0[ 2 ] );
            PB = KThreeVector( p1[ 0 ], p1[ 1 ], p1[ 2 ] );
            PC = KThreeVector( p2[ 0 ], p2[ 1 ], p2[ 2 ] );
        }

        //the line pointing along v is the y-axis
        V = PC - PB;
        Y = V.Unit();

        //q is closest point to fP[0] on line connecting fP[1] to fP[2]
        double t = (PA.Dot( V ) - PB.Dot( V )) / (V.Dot( V ));
        Q = PB + t * V;

        //the line going from fP[0] to fQ is the x-axis
        X = Q - PA;
        //gram-schmidt out any y-axis component in the x-axis
        double proj = X.Dot( Y );
        SUB = proj * Y;
        X = X - SUB;
        double H = X.Magnitude(); //compute triangle height along x

        //compute the triangles aspect ratio
        double ratio = max / H;

        return ratio;
    }

    void KGMeshTriangle::Transform( const KTransformation& transform )
    {
        transform.Apply( fP0 );
        transform.ApplyRotation( fN1 );
        transform.ApplyRotation( fN2 );
    }

    double
    KGMeshTriangle::NearestDistance(const KThreeVector& aPoint) const
    {
        KThreeVector nearest = NearestPoint(aPoint);
        return (aPoint - nearest).Magnitude();
    }

    KThreeVector
    KGMeshTriangle::NearestPoint(const KThreeVector& aPoint) const
    {
        //first construct the coordinate system for plane defined by triangle
        //origin is defined to be at p0
        KThreeVector z_axis = (fN1.Cross(fN2) ).Unit();
        KThreeVector y_axis = z_axis.Cross(fN1);

        //triangle properties
        double height = fB*fN2.Dot(y_axis);
        double p2_x = fB*fN2.Dot(fN1);

        //2d proxies of triangle points
        KThreeVector p0_proxy(0, 0, 0);
        KThreeVector p1_proxy(fA, 0, 0);
        KThreeVector p2_proxy(p2_x, height, 0);

        //project the point into the plane of the triangle
        KThreeVector del = aPoint - fP0;
        KThreeVector proj( del.Dot(fN1), del.Dot(y_axis), 0);

        if( SameSide(proj, p0_proxy, p1_proxy, p2_proxy) &&
            SameSide(proj, p1_proxy, p2_proxy, p0_proxy) &&
            SameSide(proj, p2_proxy, p0_proxy, p1_proxy) )
        {
            //projection onto triangle is the closest point
            return fP0 + (proj.X())*fN1 + (proj.Y())*y_axis;
        }

        //made it here, the nearest point on the plane is not inside the triangle
        //calculate distance to edges
        KThreeVector ab = NearestPointOnLineSegment(fP0, fP0 + fA*fN1, aPoint);
        double dist_ab =  (ab - aPoint).Magnitude();
        KThreeVector bc = NearestPointOnLineSegment(fP0 + fA*fN1, fP0 + fB*fN2, aPoint);
        double dist_bc =  (bc - aPoint).Magnitude();
        KThreeVector ca = NearestPointOnLineSegment(fP0 + fB*fN2, fP0, aPoint);
        double dist_ca =  (ca - aPoint).Magnitude();

        if( dist_ab < dist_bc && dist_ab < dist_ca)
        {
            return ab;
        }

        if( dist_bc < dist_ab && dist_bc < dist_ca)
        {
            return bc;
        }

        return ca;
    }

    KThreeVector
    KGMeshTriangle::NearestNormal(const KThreeVector& /*aPoint*/) const
    {
        return (fN1.Cross(fN2)).Unit();
    }

    bool
    KGMeshTriangle::NearestIntersection(const KThreeVector& aStart, const KThreeVector& anEnd, KThreeVector& anIntersection) const
    {
        //we compute the intersection with the plane associated with this triangle

        //first construct the coordinate system for plane defined by triangle
        //origin is defined to be at p0
        KThreeVector z_axis = (fN1.Cross(fN2) ).Unit();
        KThreeVector y_axis = z_axis.Cross(fN1);

        KThreeVector v = (anEnd - aStart);
        double len = v.Magnitude();
        v = v.Unit();
        double ndotv = z_axis.Dot(v);

        if( fabs(ndotv) < TRIANGLE_EPS )
        {
            //The line segment is ~parallel to the plane, note the line segment
            //could be in the plane, but this results in a infinite number of
            //intersections we do not check for this, since with floating point
            //math this is pretty unlikely.
            return false;
        }

        double t = ( z_axis.Dot(fP0) - z_axis.Dot(aStart) )/ndotv;

        if( t < 0)
        {
            //plane is behind the first point of the line segment
            //no intersection
            return false;
        }

        if( t > len )
        {
            //intersection is beyond the last point of the line
            //segment, so no intersection
            return false;
        }


        //passed simple checks, so there is a possible intersection given by;
        KThreeVector possible_inter = aStart + t*v;

        //project the possible interesction onto the triangle
        // KThreeVector del = possible_inter - fP0;
        // KTwoVector projection(del.Dot(fN1), del.Dot(y_axis));
        // KTwoVector proj_prime = projection - KTwoVector(fA, 0);

        // double eps = TRIANGLE_EPS*std::sqrt(fA*fB);
        // double neps = -1.0*eps;

        // double height = fB*fN2.Dot(y_axis);
        // double p2_x = fB*fN2.Dot(fN1);
        // double max_x = p2_x; if(fA > max_x){max_x = fA;};

        //triangle properties
        double height = fB*fN2.Dot(y_axis);
        double p2_x = fB*fN2.Dot(fN1);

        //2d proxies of triangle points
        KThreeVector p0_proxy(0, 0, 0);
        KThreeVector p1_proxy(fA, 0, 0);
        KThreeVector p2_proxy(p2_x, height, 0);

        //project the point into the plane of the triangle
        KThreeVector del = possible_inter - fP0;
        KThreeVector proj( del.Dot(fN1), del.Dot(y_axis), 0);

        if( SameSide(proj, p0_proxy, p1_proxy, p2_proxy) &&
            SameSide(proj, p1_proxy, p2_proxy, p0_proxy) &&
            SameSide(proj, p2_proxy, p0_proxy, p1_proxy) )
        {
            //have intersection in triangle
            anIntersection = possible_inter;
            return true;
        }

        return false;
    }

    KGPointCloud<KGMESH_DIM>
    KGMeshTriangle::GetPointCloud() const
    {
        KGPointCloud<KGMESH_DIM> point_cloud;
        point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0) );
        point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0 + fA*fN1) );
        point_cloud.AddPoint(KGPoint<KGMESH_DIM>(fP0 + fB*fN2) );
        return point_cloud;
    }

    void
    KGMeshTriangle::GetEdge(KThreeVector& start, KThreeVector& end, unsigned int index) const
    {
        if(index == 0)
        {
            start = fP0;
            end = fP0 + fA*fN1;
            return;
        }

        if(index == 1)
        {
            start = fP0 + fA*fN1;
            end = fP0 + fB*fN2;
        }

        if(index == 2)
        {
            start = fP0 + fB*fN2;
            end = fP0;
        }
    }


    KThreeVector
    KGMeshTriangle::NearestPointOnLineSegment(const KThreeVector& a, const KThreeVector& b, const KThreeVector& point) const
    {
        KThreeVector diff = b - a;
        double t = ((point - a)*diff);
        if(t < 0.){return a;};
        if(t > 1.){return b;};
        return a + t*diff;
    }


    bool KGMeshTriangle::SameSide(KThreeVector point, KThreeVector A, KThreeVector B, KThreeVector C) const
    {
        KThreeVector cp1 = (B-A).Cross(point-A);
        KThreeVector cp2 = (B-A).Cross(C-A);
        if (cp1.Dot(cp2) > 0)
        {
            return true;
        }
        return false;
    }

}
