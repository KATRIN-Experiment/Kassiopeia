#include "KGRotatedLineSegmentSpaceMesher.hh"

namespace KGeoBag
{

    KGRotatedLineSegmentSpaceMesher::KGRotatedLineSegmentSpaceMesher() :
            KGSimpleMesher()
    {
    }
    KGRotatedLineSegmentSpaceMesher::~KGRotatedLineSegmentSpaceMesher()
    {
    }

    void KGRotatedLineSegmentSpaceMesher::VisitRotatedOpenPathSpace( KGRotatedLineSegmentSpace* aRotatedLineSegmentSpace )
    {
        //create line segment points
        OpenPoints tLineSegmentPoints;
        LineSegmentToOpenPoints( aRotatedLineSegmentSpace->Path().operator ->(), tLineSegmentPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tLineSegmentPoints, aRotatedLineSegmentSpace->RotatedMeshCount(), tMeshPoints );

        //make room for ends
        tMeshPoints.fData.pop_front();
        tMeshPoints.fData.pop_back();

        //surgery
        KThreeVector tStartApex;
        if( aRotatedLineSegmentSpace->StartPath().Null() == false )
        {
            //create start circle points
            ClosedPoints tStartCirclePoints;
            CircleToClosedPoints( aRotatedLineSegmentSpace->StartPath().operator ->(), tStartCirclePoints );

            //create start flattened mesh points
            TubeMesh tStartMeshPoints;
            ClosedPointsFlattenedToTubeMeshAndApex( tStartCirclePoints, aRotatedLineSegmentSpace->StartPath()->Centroid(), aRotatedLineSegmentSpace->Path()->Start().X(), aRotatedLineSegmentSpace->FlattenedMeshCount(), aRotatedLineSegmentSpace->FlattenedMeshPower(), tStartMeshPoints, tStartApex );

            //stitch circle mesh onto main mesh
            TubeMesh::SetIt tCircleIt = tStartMeshPoints.fData.begin();
            while( tCircleIt != tStartMeshPoints.fData.end() )
            {
                tMeshPoints.fData.push_front( *tCircleIt );
                ++tCircleIt;
            }
        }
        else
        {
            //otherwise make the apex by hand
            tStartApex.SetComponents( 0., 0., aRotatedLineSegmentSpace->Path()->Start().X() );
        }

        KThreeVector tEndApex;
        if( aRotatedLineSegmentSpace->EndPath().Null() == false )
        {
            //create end circle points
            ClosedPoints tEndCirclePoints;
            CircleToClosedPoints( aRotatedLineSegmentSpace->EndPath().operator ->(), tEndCirclePoints );

            //create end flattened mesh points
            TubeMesh tEndMeshPoints;
            ClosedPointsFlattenedToTubeMeshAndApex( tEndCirclePoints, aRotatedLineSegmentSpace->EndPath()->Centroid(), aRotatedLineSegmentSpace->Path()->End().X(), aRotatedLineSegmentSpace->FlattenedMeshCount(), aRotatedLineSegmentSpace->FlattenedMeshPower(), tEndMeshPoints, tEndApex );

            TubeMesh::SetIt tCircleIt = tEndMeshPoints.fData.begin();
            while( tCircleIt != tEndMeshPoints.fData.end() )
            {
                tMeshPoints.fData.push_back( *tCircleIt );
                ++tCircleIt;
            }
        }
        else
        {
            //otherwise make the apex by hand
            tEndApex.SetComponents( 0., 0., aRotatedLineSegmentSpace->Path()->End().X() );
        }

        //lay triangles on the mesh
        TubeMeshToTriangles( tStartApex, tMeshPoints, tEndApex );

        return;
    }

}
