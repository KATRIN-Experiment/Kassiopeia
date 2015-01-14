#include "KGCylinderMesher.hh"

#include "KGMeshRectangle.hh"

namespace KGeoBag
{
    void KGCylinderMesher::VisitCylinder( KGCylinder* cylinder )
    {
        KThreeVector p0;
        KThreeVector p1;
        KThreeVector p2;
        KThreeVector p3;

        double th = 0;
        double th_last = 0;

        std::vector< double > dz( cylinder->GetLongitudinalMeshCount(), 0 );

        for( unsigned int i = 1; i <= cylinder->GetAxialMeshCount(); i++ )
        {
            th = ((double) (i % cylinder->GetAxialMeshCount())) / cylinder->GetAxialMeshCount() * 2. * M_PI;

            p0[ 0 ] = p3[ 0 ] = cylinder->GetRadius() * cos( th_last );
            p0[ 1 ] = p3[ 1 ] = -cylinder->GetRadius() * sin( th_last );
            p1[ 0 ] = p2[ 0 ] = cylinder->GetRadius() * cos( th );
            p1[ 1 ] = p2[ 1 ] = -cylinder->GetRadius() * sin( th );

            DiscretizeInterval( (cylinder->GetP1()[ 2 ] - cylinder->GetP0()[ 2 ]), cylinder->GetLongitudinalMeshCount(), cylinder->GetLongitudinalMeshPower(), dz );

            p0[ 2 ] = p1[ 2 ] = cylinder->GetP0()[ 2 ];
            for( unsigned int k = 0; k < cylinder->GetLongitudinalMeshCount(); k++ )
            {
                p2[ 2 ] = p3[ 2 ] = p0[ 2 ] + dz[ k ];

                KGMeshRectangle* r = new KGMeshRectangle( p0, p1, p2, p3 );
                AddElement( r );

                p0[ 2 ] = p1[ 2 ] = p2[ 2 ];
            }
            th_last = th;
        }
    }
}
