#include "KGRodSurfaceMesher.hh"

#include "KRotation.hh"

#include "KGMeshTriangle.hh"
#include "KGMeshRectangle.hh"

#include "KGBeamSurface.hh"

namespace KGeoBag
{
    void KGRodSurfaceMesher::VisitWrappedSurface( KGWrappedSurface< KGRod >* rodSurface )
    {
        KSmartPointer< KGRod > rod = rodSurface->GetObject();

        // First, we compute the total length of the rod
        double total_length = 0;
        for( unsigned int i = 0; i < rod->GetNCoordinates() - 1; i++ )
            total_length += sqrt( (rod->GetCoordinate( i, 0 ) - rod->GetCoordinate( i + 1, 0 )) * (rod->GetCoordinate( i, 0 ) - rod->GetCoordinate( i + 1, 0 )) + (rod->GetCoordinate( i, 1 ) - rod->GetCoordinate( i + 1, 1 )) * (rod->GetCoordinate( i, 1 ) - rod->GetCoordinate( i + 1, 1 )) + (rod->GetCoordinate( i, 2 ) - rod->GetCoordinate( i + 1, 2 )) * (rod->GetCoordinate( i, 2 ) - rod->GetCoordinate( i + 1, 2 )) );

        unsigned int disc_long_i;
        double len_i;

        double norm1[ 3 ], norm2[ 3 ];
        double normal[ 3 ];

        double tmp1[ 3 ], tmp2[ 3 ], tmp3[ 3 ], tmp4[ 3 ];
        double p1[ 3 ], p2[ 3 ], p3[ 3 ], p4[ 3 ];

        for( int j = 0; j < rod->GetNDiscRad(); j++ )
        {
            Normalize( rod->GetCoordinate( 0 ), rod->GetCoordinate( 1 ), norm1 );
            GetNormal( rod->GetCoordinate( 0 ), rod->GetCoordinate( 1 ), NULL, normal );

            KThreeVector axis( rod->GetCoordinate( 1, 0 ) - rod->GetCoordinate( 0, 0 ), rod->GetCoordinate( 1, 1 ) - rod->GetCoordinate( 0, 1 ), rod->GetCoordinate( 1, 2 ) - rod->GetCoordinate( 0, 2 ) );

            KRotation axisAngle;
            axisAngle.SetAxisAngle( axis.Unit(), 2. * M_PI / rod->GetNDiscRad() );

            KThreeVector n( normal );
            n = n.Cross( axis ).Unit();

            KThreeVector coord1 = n * rod->GetRadius();
            KThreeVector coord2 = axisAngle * coord1;

            for( int i = 0; i < j; i++ )
            {
                coord1 = axisAngle * coord1;
                coord2 = axisAngle * coord2;
            }

            p1[ 0 ] = tmp1[ 0 ] = coord1.X() + rod->GetCoordinate( 0, 0 );
            p1[ 1 ] = tmp1[ 1 ] = coord1.Y() + rod->GetCoordinate( 0, 1 );
            p1[ 2 ] = tmp1[ 2 ] = coord1.Z() + rod->GetCoordinate( 0, 2 );
            p2[ 0 ] = tmp2[ 0 ] = coord2.X() + rod->GetCoordinate( 0, 0 );
            p2[ 1 ] = tmp2[ 1 ] = coord2.Y() + rod->GetCoordinate( 0, 1 );
            p2[ 2 ] = tmp2[ 2 ] = coord2.Z() + rod->GetCoordinate( 0, 2 );

            for( unsigned int i = 0; i < rod->GetNCoordinates() - 1; i++ )
            {
                // the longititudinal discretization of the rod segment is a fraction of
                // the user-defined longitudinal discretization
                len_i = sqrt( (rod->GetCoordinate( i, 0 ) - rod->GetCoordinate( i + 1, 0 )) * (rod->GetCoordinate( i, 0 ) - rod->GetCoordinate( i + 1, 0 )) + (rod->GetCoordinate( i, 1 ) - rod->GetCoordinate( i + 1, 1 )) * (rod->GetCoordinate( i, 1 ) - rod->GetCoordinate( i + 1, 1 )) + (rod->GetCoordinate( i, 2 ) - rod->GetCoordinate( i + 1, 2 )) * (rod->GetCoordinate( i, 2 ) - rod->GetCoordinate( i + 1, 2 )) );
                disc_long_i = rod->GetNDiscLong() * (len_i / total_length);
                if( disc_long_i < 1 )
                    disc_long_i = 1;

                // now, we determine the planes that constrict rod segment i from the
                // i+1-end (the plane on the i-th end were computed during the last pass)
                Normalize( rod->GetCoordinate( i ), rod->GetCoordinate( i + 1 ), norm2 );

                for( unsigned int k = 0; k < 3; k++ )
                {
                    tmp3[ k ] = p2[ k ] + norm2[ k ];
                    tmp4[ k ] = p1[ k ] + norm2[ k ];
                }

                if( i != rod->GetNCoordinates() - 2 )
                {
                    double tmp_norm[ 3 ];
                    Normalize( rod->GetCoordinate( i + 1 ), rod->GetCoordinate( i + 2 ), tmp_norm );
                    double len = 0.;
                    for( unsigned int k = 0; k < 3; k++ )
                    {
                        norm2[ k ] = .5 * (norm2[ k ] + tmp_norm[ k ]);
                        len += norm2[ k ] * norm2[ k ];
                    }
                    len = sqrt( len );
                    for( unsigned int k = 0; k < 3; k++ )
                        norm2[ k ] /= len;
                }

                KGBeam::LinePlaneIntersection( tmp4, p1, rod->GetCoordinate( i + 1 ), norm2, p4 );
                KGBeam::LinePlaneIntersection( p2, tmp3, rod->GetCoordinate( i + 1 ), norm2, p3 );

                AddTrapezoid( p1, p2, p3, p4, disc_long_i );

                for( unsigned int k = 0; k < 3; k++ )
                {
                    p1[ k ] = p4[ k ];
                    p2[ k ] = p3[ k ];
                    norm1[ k ] = norm2[ k ];
                }
            }
        }
    }

    //______________________________________________________________________________

    void KGRodSurfaceMesher::Normalize( const double* p1, const double* p2, double* norm )
    {
        double len = 0.;
        for( unsigned int i = 0; i < 3; i++ )
        {
            norm[ i ] = p2[ i ] - p1[ i ];
            len += norm[ i ] * norm[ i ];
        }
        len = sqrt( len );
        for( unsigned int i = 0; i < 3; i++ )
            norm[ i ] /= len;
    }

    //______________________________________________________________________________

    void KGRodSurfaceMesher::GetNormal( const double* p1, const double* p2, const double* oldNormal, double* normal )
    {
        // Given a line going through p1 and p2, returns a unit vector that lies in
        // the plane normal to the line.

        // we start by constructing the unit normal vector pointing from n1 to n2
        double n[ 3 ];
        double len = 0;
        for( unsigned int i = 0; i < 3; i++ )
        {
            n[ i ] = p2[ i ] - p1[ i ];
            len += n[ i ] * n[ i ];
        }
        len = sqrt( len );
        for( unsigned int i = 0; i < 3; i++ )
            n[ i ] /= len;

        if( oldNormal == NULL )
        {
            // we then start with a normal vector whose sole component lies in the
            // direction of the smallest magnitude of n
            int iSmallest = 0;
            double smallest = 2.;
            for( unsigned int i = 0; i < 3; i++ )
            {
                if( smallest > fabs( n[ i ] ) )
                {
                    smallest = fabs( n[ i ] );
                    iSmallest = i;
                }
            }
            normal[ 0 ] = normal[ 1 ] = normal[ 2 ] = 0;
            normal[ iSmallest ] = 1.;
        }
        else
        {
            for( unsigned int i = 0; i < 3; i++ )
                normal[ i ] = oldNormal[ i ];
        }

        // we then subtract away the parts of normal that are in the direction of n
        double ndotnormal = 0.;
        for( unsigned int i = 0; i < 3; i++ )
            ndotnormal += n[ i ] * normal[ i ];

        if( fabs( fabs( ndotnormal ) - 1. ) < 1.e-8 )
        {
            double tmp = normal[ 0 ];
            normal[ 0 ] = normal[ 1 ];
            normal[ 1 ] = normal[ 2 ];
            normal[ 2 ] = tmp;
            for( unsigned int i = 0; i < 3; i++ )
                ndotnormal += n[ i ] * normal[ i ];
        }

        len = 0.;
        for( unsigned int i = 0; i < 3; i++ )
        {
            normal[ i ] -= ndotnormal * n[ i ];
            len += normal[ i ] * normal[ i ];
        }
        len = sqrt( len );
        for( unsigned int i = 0; i < 3; i++ )
            normal[ i ] /= len;

        return;
    }

    //______________________________________________________________________________

    void KGRodSurfaceMesher::AddTrapezoid( const double* P1, const double* P2, const double* P3, const double* P4, const int nDisc )
    {
        // Adds a long, thin trapezoid defined with the short segments p1-p2 & p3-p4
        // and parallel segments p1-p4 & p2-p3.

        KThreeVector p1, p2, p3, p4;
        KThreeVector n12, n14, n23, n34;
        double len12, len14, len23, len34;
        len12 = len14 = len23 = len34 = 0.;

        for( unsigned int j = 0; j < 3; j++ )
        {
            /// Corners...
            p1[ j ] = P1[ j ];
            p2[ j ] = P2[ j ];
            p3[ j ] = P3[ j ];
            p4[ j ] = P4[ j ];

            // ... unit vectors...
            n12[ j ] = p2[ j ] - p1[ j ];
            n14[ j ] = p4[ j ] - p1[ j ];
            n23[ j ] = p3[ j ] - p2[ j ];
            n34[ j ] = p4[ j ] - p3[ j ];

            // ... and magnitudes.
            len12 += n12[ j ] * n12[ j ];
            len14 += n14[ j ] * n14[ j ];
            len23 += n23[ j ] * n23[ j ];
            len34 += n34[ j ] * n34[ j ];
        }
        len12 = sqrt( len12 );
        len14 = sqrt( len14 );
        len23 = sqrt( len23 );
        len34 = sqrt( len34 );

        for( unsigned int j = 0; j < 3; j++ )
        {
            n12[ j ] /= len12;
            n14[ j ] /= len14;
            n23[ j ] /= len23;
            n34[ j ] /= len34;
        }

        // we now have a trapezoid defined by p1,p2,p3,p4

        double n12dotn14 = n12[ 0 ] * n14[ 0 ] + n12[ 1 ] * n14[ 1 ] + n12[ 2 ] * n14[ 2 ];
        double theta214 = acos( n12dotn14 );

        // ratio: the ratio of the two sides of the triangle at either end of the
        // trapezoid.  If the ratio is less than the value defined below, two
        // triangles are made whose values are derived using this value.
        double ratio = .7;

        if( fabs( theta214 - M_PI / 2. ) > 1.e-4 )
        {
            // the start is not rectangular

            KGMeshTriangle* t;
            KThreeVector tmp;

            if( theta214 - M_PI / 2. > 0. )
            {
                for( unsigned int j = 0; j < 3; j++ )
                    tmp[ j ] = p2[ j ] + n23[ j ] * len12 * sin( theta214 - M_PI / 2. );

                // If the ratio of the lengths of the triangle sides is greater than a
                // fixed value, we only need one triangle
                if( (tmp - p1).Magnitude() / (tmp - p2).Magnitude() > ratio )
                {
                    t = new KGMeshTriangle( tmp, p1, p2 );
                    AddElement( t );

                    p2 = tmp;
                }
                // Otherwise, we need to use two triangles to ensure that we aren't
                // creating disproportionately narrow triangles
                else
                {
                    double newLen = (len12 * ratio < len14 ? len12 * ratio : len14);
                    tmp = p2 + n23 * newLen;
                    t = new KGMeshTriangle( tmp, p1, p2 );
                    AddElement( t );

                    for( unsigned int j = 0; j < 3; j++ )
                    {
                        p2[ j ] = tmp[ j ];
                        tmp[ j ] = p1[ j ] + n14[ j ] * ((tmp - p1).Magnitude() - len12 * sin( theta214 - M_PI / 2. ));
                    }

                    t = new KGMeshTriangle( tmp, p1, p2 );
                    AddElement( t );

                    for( unsigned int j = 0; j < 3; j++ )
                        p1[ j ] = tmp[ j ];
                }
            }
            else
            {
                for( unsigned int j = 0; j < 3; j++ )
                    tmp[ j ] = p1[ j ] + n14[ j ] * len12 * sin( M_PI / 2. - theta214 );

                // If the ratio of the lengths of the triangle sides is greater than a
                // fixed value, we only need one triangle
                if( (p1 - p2).Magnitude() / (p1 - tmp).Magnitude() > ratio )
                {
                    t = new KGMeshTriangle( p1, p2, tmp );
                    AddElement( t );

                    p1 = tmp;
                }
                else
                {
                    double newLen = (len12 * ratio < len23 ? len12 * ratio : len23);
                    tmp = p1 + n14 * newLen;
                    t = new KGMeshTriangle( tmp, p2, p1 );
                    AddElement( t );

                    for( unsigned int j = 0; j < 3; j++ )
                    {
                        p1[ j ] = tmp[ j ];
                        tmp[ j ] = p2[ j ] + n23[ j ] * ((p1 - tmp).Magnitude() - len12 * sin( M_PI / 2. - theta214 ));
                    }

                    t = new KGMeshTriangle( tmp, p2, p1 );
                    AddElement( t );

                    p2 = tmp;
                }
            }
        }

        double n43dotn41 = n34[ 0 ] * n14[ 0 ] + n34[ 1 ] * n14[ 1 ] + n34[ 2 ] * n14[ 2 ];
        double theta341 = acos( n43dotn41 );

        if( fabs( theta341 - M_PI / 2. ) > 1.e-4 )
        {
            // the end is not rectangular

            KGMeshTriangle* t = 0;
            KThreeVector tmp;

            if( theta341 - M_PI / 2. > 0. )
            {
                for( unsigned int j = 0; j < 3; j++ )
                    tmp[ j ] = p3[ j ] - n23[ j ] * len34 * sin( theta341 - M_PI / 2. );

                // If the ratio of the lengths of the triangle sides is greater than a
                // fixed value, we only need one triangle
                if( (tmp - p4).Magnitude() / (tmp - p3).Magnitude() > ratio )
                {
                    t = new KGMeshTriangle( tmp, p4, p3 );
                    AddElement( t );

                    p3 = tmp;
                }
                // Otherwise, we need to use two triangles to ensure that we aren't
                // creating disproportionately narrow triangles
                else
                {
                    double newLen = (len34 * ratio < len23 ? len34 * ratio : len23);
                    tmp = p3 - n23 * newLen;
                    t = new KGMeshTriangle( tmp, p4, p3 );
                    AddElement( t );

                    for( unsigned int j = 0; j < 3; j++ )
                    {
                        p3[ j ] = tmp[ j ];
                        tmp[ j ] = p4[ j ] - n14[ j ] * ((tmp - p3).Magnitude() - len34 * sin( theta341 - M_PI / 2. ));
                    }

                    t = new KGMeshTriangle( tmp, p4, p3 );
                    AddElement( t );

                    p4 = tmp;
                }
            }
            else
            {
                tmp = p4 - n14 * len34 * sin( M_PI / 2. - theta341 );

                // If the ratio of the lengths of the triangle sides is greater than a
                // fixed value, we only need one triangle
                if( (tmp - p3).Magnitude() / (tmp - p4).Magnitude() > ratio )
                {
                    t = new KGMeshTriangle( tmp, p3, p4 );
                    AddElement( t );

                    p4 = tmp;
                }
                else
                {
                    double newLen = (len34 * ratio < len14 ? len34 * ratio : len14);
                    tmp = p4 - n14 * newLen;
                    t = new KGMeshTriangle( tmp, p3, p4 );
                    AddElement( t );

                    for( unsigned int j = 0; j < 3; j++ )
                    {
                        p4[ j ] = tmp[ j ];
                        tmp[ j ] = p3[ j ] - n23[ j ] * ((tmp - p4).Magnitude() - len34 * sin( M_PI / 2. - theta341 ));
                    }

                    t = new KGMeshTriangle( tmp, p3, p4 );
                    AddElement( t );

                    p3 = tmp;
                }
            }
        }

        double d1 = 0.;
        double d2 = 0.;
        KThreeVector n1;
        KThreeVector n2;

        for( unsigned int j = 0; j < 3; j++ )
        {
            n1[ j ] = p4[ j ] - p1[ j ];
            n2[ j ] = p2[ j ] - p1[ j ];

            d1 += n1[ j ] * n1[ j ];
            d2 += n2[ j ] * n2[ j ];
        }
        d1 = sqrt( d1 );
        d2 = sqrt( d2 );

        for( unsigned int j = 0; j < 3; j++ )
        {
            n1[ j ] /= d1;
            n2[ j ] /= d2;
        }

        KGMeshRectangle* r = new KGMeshRectangle( d1, d2, p1, n1, n2 );
        RefineAndAddElement( r, nDisc, 2, 1, 1 );
    }
}
