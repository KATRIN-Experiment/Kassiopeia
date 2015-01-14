#include "KGMeshTriangle.hh"

#include <math.h>

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

}
