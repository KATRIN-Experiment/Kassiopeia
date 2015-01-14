#include "KGComplexMesher.hh"

#include "KGCoreMessage.hh"

#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"

namespace KGeoBag
{

    KGComplexMesher::KGComplexMesher() :
        KGMesherBase()
    {
    }
    KGComplexMesher::~KGComplexMesher()
    {
    }

    void KGComplexMesher::AddElement( KGMeshElement* e )
    {
        fCurrentElements->push_back( e );
        return;
    }

    void KGComplexMesher::DiscretizeInterval( double interval, int nSegments, double power, std::vector< double >& segments )
    {
        if( nSegments == 1 )
            segments[ 0 ] = interval;
        else
        {
            double inc1, inc2;
            double mid = interval * .5;
            if( nSegments % 2 == 1 )
            {
                segments[ nSegments / 2 ] = interval / nSegments;
                mid -= interval / (2 * nSegments);
            }

            for( int i = 0; i < nSegments / 2; i++ )
            {
                inc1 = ((double) i) / (nSegments / 2);
                inc2 = ((double) (i + 1)) / (nSegments / 2);

                inc1 = pow( inc1, power );
                inc2 = pow( inc2, power );

                segments[ i ] = segments[ nSegments - (i + 1) ] = mid * (inc2 - inc1);
            }
        }
        return;
    }

    void KGComplexMesher::RefineAndAddElement( KGMeshRectangle* rectangle, int nElements_A, double power_A, int nElements_B, double power_B )
    {
        if( nElements_A == 0 || nElements_B == 0 )
        {
            AddElement( rectangle );
            return;
        }

        // vectors a,b contain the lengths of the rectangles
        std::vector< double > a( nElements_A );
        std::vector< double > b( nElements_B );

        DiscretizeInterval( rectangle->GetA(), nElements_A, power_A, a );
        DiscretizeInterval( rectangle->GetB(), nElements_B, power_B, b );

        // dA and db are the offsets for each fP0[3] cornerpoint
        double dA = 0;
        double dB = 0;

        double a_new;
        double b_new;
        double p0_new[ 3 ];

        for( int i = 0; i < nElements_A; i++ )
        {
            // set length A
            a_new = a[ i ];
            dB = 0;
            for( int j = 0; j < nElements_B; j++ )
            {
                // set P0
                for( int k = 0; k < 3; k++ )
                    p0_new[ k ] = (rectangle->GetP0()[ k ] + dA * rectangle->GetN1()[ k ] + dB * rectangle->GetN2()[ k ]);
                // set length B
                b_new = b[ j ];
                dB += b[ j ];

                // add r to the vector
                KGMeshRectangle* newRectangle = new KGMeshRectangle( a_new, b_new, p0_new, rectangle->GetN1(), rectangle->GetN2() );
                AddElement( newRectangle );
            }
            dA += a[ i ];
        }

        delete rectangle;
        return;
    }

    void KGComplexMesher::RefineAndAddElement( KGMeshTriangle* triangle, int nElements, double power )
    {
        if( nElements == 0 )
        {
            AddElement( triangle );
            return;
        }

        // vectors a,b contain the lengths of the triangles
        std::vector< double > a( nElements );
        std::vector< double > b( nElements );

        DiscretizeInterval( triangle->GetA(), nElements, power, a );
        DiscretizeInterval( triangle->GetB(), nElements, power, b );

        double P0[ 3 ];
        double P1[ 3 ];
        double P2[ 3 ];

        // loop over the A dimension (n1)
        for( int i = 0; i < nElements; i++ )
        {
            // initialize the first triangle in the column
            for( int k = 0; k < 3; k++ )
            {
                P0[ k ] = triangle->GetP0()[ k ];

                for( int m = 0; m < i; m++ )
                    P0[ k ] += a.at( m ) * triangle->GetN1()[ k ];
                P1[ k ] = P0[ k ] + a.at( i ) * triangle->GetN1()[ k ];
                P2[ k ] = P0[ k ] + b.at( 0 ) * triangle->GetN2()[ k ];
            }

            // loop over the B dimension (n2)
            for( int j = 0; j < nElements - i; j++ )
            {
                KGMeshTriangle* newTriangle = new KGMeshTriangle( P0, P1, P2 );
                AddElement( newTriangle );

                // if we are not at the top of the column, there is an inverted triangle
                // too.
                if( j != nElements - i - 1 )
                {
                    for( int k = 0; k < 3; k++ )
                        P0[ k ] += a.at( i ) * triangle->GetN1()[ k ] + b.at( j ) * triangle->GetN2()[ k ];

                    KGMeshTriangle* newTriangle2 = new KGMeshTriangle( P0, P1, P2 );
                    AddElement( newTriangle2 );

                    for( int k = 0; k < 3; k++ )
                    {
                        P1[ k ] = P0[ k ];
                        P0[ k ] = P2[ k ];
                        P2[ k ] += b.at( j + 1 ) * triangle->GetN2()[ k ];
                    }
                }
            }
        }

        delete triangle;
        return;
    }

    void KGComplexMesher::RefineAndAddElement( KGMeshWire* wire, int nElements, double power )
    {
        if( nElements == 0 )
        {
            AddElement( wire );
            return;
        }

        double A[ 3 ] =
        { 0., 0., 0. }; // new wire parameters
        double B[ 3 ] =
        { 0., 0., 0. };

        std::vector< std::vector< double > > inc( 3, std::vector< double >( nElements, 0 ) );

        DiscretizeInterval( (wire->GetP1()[ 0 ] - wire->GetP0()[ 0 ]), nElements, power, inc[ 0 ] );
        DiscretizeInterval( (wire->GetP1()[ 1 ] - wire->GetP0()[ 1 ]), nElements, power, inc[ 1 ] );
        DiscretizeInterval( (wire->GetP1()[ 2 ] - wire->GetP0()[ 2 ]), nElements, power, inc[ 2 ] );

        for( int i = 0; i < 3; i++ )
            B[ i ] = wire->GetP0()[ i ];

        for( int i = 0; i < nElements / 2; i++ )
        {
            for( int j = 0; j < 3; j++ )
            {
                A[ j ] = B[ j ];
                B[ j ] += inc[ j ][ i ];
            }
            KGMeshWire* newWire = new KGMeshWire( A, B, wire->GetDiameter() );
            AddElement( newWire );
        }

        for( int i = 0; i < 3; i++ )
            A[ i ] = wire->GetP1()[ i ];
        for( int i = nElements - 1; i >= nElements / 2; i-- )
        {
            for( int j = 0; j < 3; j++ )
            {
                B[ j ] = A[ j ];
                A[ j ] -= inc[ j ][ i ];
            }
            KGMeshWire* newWire = new KGMeshWire( A, B, wire->GetDiameter() );
            AddElement( newWire );
        }

        delete wire;
        return;
    }

}
