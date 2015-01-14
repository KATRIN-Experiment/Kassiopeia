#include "KGExtrudedPolyLoopSpaceMesher.hh"

namespace KGeoBag
{

    KGExtrudedPolyLoopSpaceMesher::KGExtrudedPolyLoopSpaceMesher() :
            KGSimpleMesher()
    {
    }
    KGExtrudedPolyLoopSpaceMesher::~KGExtrudedPolyLoopSpaceMesher()
    {
    }

    void KGExtrudedPolyLoopSpaceMesher::VisitExtrudedClosedPathSpace( KGExtrudedPolyLoopSpace* aExtrudedPolyLoopSpace )
    {
        //create circle points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aExtrudedPolyLoopSpace->Path().operator->(), tPolyLoopPoints );

        //create extruded points
        TubeMesh tMeshPoints;
        ClosedPointsExtrudedToTubeMesh( tPolyLoopPoints, aExtrudedPolyLoopSpace->ZMin(), aExtrudedPolyLoopSpace->ZMax(), aExtrudedPolyLoopSpace->ExtrudedMeshCount(), aExtrudedPolyLoopSpace->ExtrudedMeshPower(), tMeshPoints );

        //create start flattened mesh points
        TubeMesh tStartMeshPoints;
        KThreeVector tStartApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tPolyLoopPoints, aExtrudedPolyLoopSpace->Path()->Centroid(), aExtrudedPolyLoopSpace->ZMin(), aExtrudedPolyLoopSpace->FlattenedMeshCount(), aExtrudedPolyLoopSpace->FlattenedMeshPower(), tStartMeshPoints, tStartApex );

        //create end flattened mesh points
        TubeMesh tEndMeshPoints;
        KThreeVector tEndApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tPolyLoopPoints, aExtrudedPolyLoopSpace->Path()->Centroid(), aExtrudedPolyLoopSpace->ZMax(), aExtrudedPolyLoopSpace->FlattenedMeshCount(), aExtrudedPolyLoopSpace->FlattenedMeshPower(), tEndMeshPoints, tEndApex );

        //surgery
        tMeshPoints.fData.pop_front();
        for( TubeMesh::SetIt tStartIt = tStartMeshPoints.fData.begin(); tStartIt != tStartMeshPoints.fData.end(); ++tStartIt )
        {
            tMeshPoints.fData.push_front( *tStartIt );
        }

        tMeshPoints.fData.pop_back();
        for( TubeMesh::SetIt tEndIt = tEndMeshPoints.fData.begin(); tEndIt != tEndMeshPoints.fData.end(); ++tEndIt )
        {
            tMeshPoints.fData.push_back( *tEndIt );
        }

        //create mesh
        TubeMeshToTriangles( tStartApex, tMeshPoints, tEndApex );

        return;
    }

}
