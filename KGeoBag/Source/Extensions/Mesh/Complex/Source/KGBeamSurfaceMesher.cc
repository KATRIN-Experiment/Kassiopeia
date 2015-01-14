#include "KGBeamSurfaceMesher.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"

namespace KGeoBag
{

    void KGBeamSurfaceMesher::VisitWrappedSurface( KGBeamSurface* beamSurface )
    {
        KSmartPointer < KGBeam > beam = beamSurface->GetObject();

        KThreeVector p1, p2, p3, p4;
        KThreeVector n12, n14, n23, n34;
        double len12, len14, len23, len34;

        for( unsigned int i = 0; i < beam->GetStartCoords().size() - 1; i++ )
        {
            len12 = len14 = len23 = len34 = 0.;

            for( unsigned int j = 0; j < 3; j++ )
            {
                // corners...
                p1[ j ] = beam->GetStartCoords().at( i ).at( j );
                p2[ j ] = beam->GetStartCoords().at( i + 1 ).at( j );
                p3[ j ] = beam->GetEndCoords().at( i + 1 ).at( j );
                p4[ j ] = beam->GetEndCoords().at( i ).at( j );

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

            if( fabs( theta214 - M_PI / 2. ) > 1.e-4 )
            {
                // the start is not rectangular

                KThreeVector tmp;

                if( theta214 - M_PI / 2. > 0. )
                {
                    for( unsigned int j = 0; j < 3; j++ )
                        tmp[ j ] = p2[ j ] + n23[ j ] * len12 * sin( theta214 - M_PI / 2. );

                    KGMeshTriangle* t = new KGMeshTriangle( p1, p2, tmp );
                    RefineAndAddElement( t, beam->GetRadialDiscretization( i ), 2 );

                    for( unsigned int j = 0; j < 3; j++ )
                        p2[ j ] = tmp[ j ];
                }
                else
                {
                    for( unsigned int j = 0; j < 3; j++ )
                        tmp[ j ] = p1[ j ] + n14[ j ] * len12 * sin( M_PI / 2. - theta214 );

                    KGMeshTriangle* t = new KGMeshTriangle( p1, p2, tmp );
                    RefineAndAddElement( t, beam->GetRadialDiscretization( i ), 2 );

                    for( unsigned int j = 0; j < 3; j++ )
                        p1[ j ] = tmp[ j ];
                }
            }

            double n43dotn41 = n34[ 0 ] * n14[ 0 ] + n34[ 1 ] * n14[ 1 ] + n34[ 2 ] * n14[ 2 ];
            double theta341 = acos( n43dotn41 );

            if( fabs( theta341 - M_PI / 2. ) > 1.e-4 )
            {
                // the end is not rectangular

                double tmp[ 3 ];

                if( theta341 - M_PI / 2. > 0. )
                {
                    for( unsigned int j = 0; j < 3; j++ )
                        tmp[ j ] = p3[ j ] - n23[ j ] * len34 * sin( theta341 - M_PI / 2. );

                    KGMeshTriangle* t = new KGMeshTriangle( p4, p3, tmp );
                    RefineAndAddElement( t, beam->GetRadialDiscretization( i ), 2 );

                    for( unsigned int j = 0; j < 3; j++ )
                        p3[ j ] = tmp[ j ];
                }
                else
                {
                    for( unsigned int j = 0; j < 3; j++ )
                        tmp[ j ] = p4[ j ] - n14[ j ] * len34 * sin( M_PI / 2. - theta341 );

                    KGMeshTriangle* t = new KGMeshTriangle( p4, p3, tmp );
                    RefineAndAddElement( t, beam->GetRadialDiscretization( i ), 2 );

                    for( unsigned int j = 0; j < 3; j++ )
                        p4[ j ] = tmp[ j ];
                }
            }

            double d1 = 0.;
            double d2 = 0.;
            double n1[ 3 ];
            double n2[ 3 ];

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
            RefineAndAddElement( r, beam->GetLongitudinalDiscretization(), 2, beam->GetRadialDiscretization( i ), 2 );
        }
    }
}
