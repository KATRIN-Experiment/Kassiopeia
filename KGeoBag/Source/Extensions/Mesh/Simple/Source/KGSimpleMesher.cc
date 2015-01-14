#include "KGSimpleMesher.hh"

#include "KGMeshMessage.hh"
#include "KGMeshTriangle.hh"

#include "KConst.h"
#include <cmath>

namespace KGeoBag
{

    KGSimpleMesher::KGSimpleMesher() :
    KGMesherBase()
    {
    }
    KGSimpleMesher::~KGSimpleMesher()
    {
    }

    //*******************
    //partition functions
    //*******************

    void KGSimpleMesher::SymmetricPartition( const double& aStart, const double& aStop, const unsigned int& aCount, const double& aPower, Partition& aPartition )
    {
        register double tPower = aPower;
        register double tStart = aStart;
        register double tStop = aStop;
        register double tMid = .5 * (tStop - tStart);
        register double tY;
        register double tX;

        aPartition.fData.clear();
        for( unsigned int tIndex = 0; tIndex <= aCount; tIndex++ )
        {
            tY = (double) (tIndex) / (double) (aCount);
            if( tY < 0.5 )
            {
                tX = tStart + tMid * pow( 2. * tY, tPower );
            }
            else
            {
                tX = tStop - tMid * pow( 2. - 2. * tY, tPower );
            }
            aPartition.fData.push_back( tX );
        }

        return;
    }
    void KGSimpleMesher::ForwardPartition( const double& aStart, const double& aStop, const unsigned int& aCount, const double& aPower, Partition& aPartition )
    {
        register double tPower = aPower;
        register double tStart = aStart;
        register double tStop = aStop;
        register double tLength = tStop - tStart;
        register double tY;
        register double tX;

        aPartition.fData.clear();
        for( unsigned int tIndex = 0; tIndex <= aCount; tIndex++ )
        {
            tY = (double) (tIndex) / (double) (aCount);
            tX = tStart + tLength * pow( tY, tPower );
            aPartition.fData.push_back( tX );
        }

        return;
    }
    void KGSimpleMesher::BackwardPartition( const double& aStart, const double& aStop, const unsigned int& aCount, const double& aPower, Partition& aPartition )
    {
        register double tPower = aPower;
        register double tStart = aStart;
        register double tStop = aStop;
        register double tLength = tStop - tStart;
        register double tY;
        register double tX;

        aPartition.fData.clear();
        for( unsigned int tIndex = 0; tIndex <= aCount; tIndex++ )
        {
            tY = (double) (tIndex) / (double) (aCount);
            tX = tStop - tLength * pow( 1. - 1. * tY, tPower );
            aPartition.fData.push_back( tX );
        }

        return;
    }

    //****************
    //points functions
    //****************

    void KGSimpleMesher::LineSegmentToOpenPoints( const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints )
    {
        aPoints.fData.clear();

        Partition tPartition;
        Partition::It tPartitionIt;

        SymmetricPartition( 0., aLineSegment->Length(), aLineSegment->MeshCount(), aLineSegment->MeshPower(), tPartition );

        for( tPartitionIt = tPartition.fData.begin(); tPartitionIt != tPartition.fData.end(); tPartitionIt++ )
        {
            aPoints.fData.push_back( aLineSegment->At( *tPartitionIt ) );
        }

        meshmsg_debug( "line segment partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }
    void KGSimpleMesher::ArcSegmentToOpenPoints( const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints )
    {
        aPoints.fData.clear();

        Partition tPartition;
        Partition::It tPartitionIt;

        SymmetricPartition( 0., anArcSegment->Length(), anArcSegment->MeshCount(), 1., tPartition );

        for( tPartitionIt = tPartition.fData.begin(); tPartitionIt != tPartition.fData.end(); tPartitionIt++ )
        {
            aPoints.fData.push_back( anArcSegment->At( *tPartitionIt ) );
        }

        meshmsg_debug( "arc segment partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }
    void KGSimpleMesher::PolyLineToOpenPoints( const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints )
    {
        aPoints.fData.clear();

        const KGPlanarPolyLine::Set& tElements = aPolyLine->Elements();
        KGPlanarPolyLine::CIt tElementIt;
        const KGPlanarOpenPath* tElement;
        const KGPlanarLineSegment* tLineSegmentElement;
        const KGPlanarArcSegment* tArcSegmentElement;

        OpenPoints tSubPoints;
        for( tElementIt = tElements.begin(); tElementIt != tElements.end(); tElementIt++ )
        {
            tElement = *tElementIt;

            tLineSegmentElement = dynamic_cast< const KGPlanarLineSegment* >( tElement );
            if( tLineSegmentElement != NULL )
            {
                LineSegmentToOpenPoints( tLineSegmentElement, tSubPoints );
                aPoints.fData.insert( aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()) );
                continue;
            }

            tArcSegmentElement = dynamic_cast< const KGPlanarArcSegment* >( tElement );
            if( tArcSegmentElement != NULL )
            {
                ArcSegmentToOpenPoints( tArcSegmentElement, tSubPoints );
                aPoints.fData.insert( aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()) );
                continue;
            }
        }

        aPoints.fData.push_back( aPolyLine->End() );

        meshmsg_debug( "poly line partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }
    void KGSimpleMesher::CircleToClosedPoints( const KGPlanarCircle* aCircle, ClosedPoints& aPoints )
    {
        aPoints.fData.clear();

        Partition tPartition;
        Partition::It tPartitionIt;

        SymmetricPartition( 0., aCircle->Length(), aCircle->MeshCount(), 1., tPartition );

        for( tPartitionIt = tPartition.fData.begin(); tPartitionIt != --(tPartition.fData.end()); tPartitionIt++ )
        {
            aPoints.fData.push_back( aCircle->At( *tPartitionIt ) );
        }

        meshmsg_debug( "circle partitioned into <" << aPoints.fData.size() << "> closed points vertices" << eom );

        return;
    }
    void KGSimpleMesher::PolyLoopToClosedPoints( const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints )
    {
        aPoints.fData.clear();

        const KGPlanarPolyLoop::Set& tElements = aPolyLoop->Elements();
        KGPlanarPolyLoop::CIt tElementIt;
        const KGPlanarOpenPath* tElement;
        const KGPlanarLineSegment* tLineSegmentElement;
        const KGPlanarArcSegment* tArcSegmentElement;

        OpenPoints tSubPoints;
        for( tElementIt = tElements.begin(); tElementIt != tElements.end(); tElementIt++ )
        {
            tElement = *tElementIt;

            tLineSegmentElement = dynamic_cast< const KGPlanarLineSegment* >( tElement );
            if( tLineSegmentElement != NULL )
            {
                LineSegmentToOpenPoints( tLineSegmentElement, tSubPoints );
                aPoints.fData.insert( aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()) );
                continue;
            }

            tArcSegmentElement = dynamic_cast< const KGPlanarArcSegment* >( tElement );
            if( tArcSegmentElement != NULL )
            {
                ArcSegmentToOpenPoints( tArcSegmentElement, tSubPoints );
                aPoints.fData.insert( aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()) );
                continue;
            }
        }

        meshmsg_debug( "poly loop partitioned into <" << aPoints.fData.size() << "> closed points vertices" << eom );

        return;
    }

    //**************
    //mesh functions
    //**************

    void KGSimpleMesher::ClosedPointsFlattenedToTubeMeshAndApex( const ClosedPoints& aPoints, const KTwoVector& aCentroid, const double& aZ, const unsigned int& aCount, const double& aPower, TubeMesh& aMesh, KThreeVector& anApex )
    {
        aMesh.fData.clear();

        Partition tPartition;
        ForwardPartition( 0., 1., aCount, aPower, tPartition );

        KThreeVector tPoint;
        TubeMesh::Group tGroup;
        for( Partition::It tPartitionIt = tPartition.fData.begin(); tPartitionIt != --(tPartition.fData.end()); tPartitionIt++ )
        {
            tGroup.clear();
            for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
            {
                tPoint.X() = (*tPointsIt).X() + (*tPartitionIt) * (aCentroid.X() - (*tPointsIt).X());
                tPoint.Y() = (*tPointsIt).Y() + (*tPartitionIt) * (aCentroid.Y() - (*tPointsIt).Y());
                tPoint.Z() = aZ;
                tGroup.push_back( tPoint );
            }
            aMesh.fData.push_back( tGroup );
        }
        anApex.X() = aCentroid.X();
        anApex.Y() = aCentroid.Y();
        anApex.Z() = aZ;

        meshmsg_debug( "flattened closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

        return;
    }
    void KGSimpleMesher::OpenPointsRotatedToTubeMesh( const OpenPoints& aPoints, const unsigned int& aCount, TubeMesh& aMesh )
    { 
        aMesh.fData.clear();

        Partition tPartition;
        SymmetricPartition( 0., 1., aCount, 1., tPartition );

        KThreeVector tPoint;
        TubeMesh::Group tGroup;
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tGroup.clear();
            for( Partition::It tPartitionIt = tPartition.fData.begin(); tPartitionIt != --(tPartition.fData.end()); tPartitionIt++ )
            {
                tPoint.X() = (*tPointsIt).Y() * cos( 2. * KConst::Pi() * (*tPartitionIt) );
                tPoint.Y() = (*tPointsIt).Y() * sin( 2. * KConst::Pi() * (*tPartitionIt) );
                tPoint.Z() = (*tPointsIt).X();
                tGroup.push_back( tPoint );
            }
            aMesh.fData.push_back( tGroup );
        }

        meshmsg_debug( "rotated open points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

        return;
    }
    void KGSimpleMesher::OpenPointsRotatedToShellMesh( const OpenPoints& aPoints, const unsigned int& aCount, const double& aPower, ShellMesh& aMesh, const double& aAngleStart, const double& aAngleStop )
    { std::cout << "building a shell mesh" << std::endl;
    aMesh.fData.clear();
    
    double tAngle = (aAngleStop - aAngleStart)/360;
    Partition tPartition;
    SymmetricPartition( 0., 1., aCount, aPower, tPartition );

    KThreeVector tPoint;
    ShellMesh::Group tGroup;
    for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
    {
        tGroup.clear();
        for( Partition::It tPartitionIt = tPartition.fData.begin(); tPartitionIt != tPartition.fData.end(); tPartitionIt++ )
        {
            tPoint.X() = (*tPointsIt).Y() * cos( 2. * KConst::Pi() * (*tPartitionIt) * tAngle + aAngleStart* KConst::Pi()/180.);
            tPoint.Y() = (*tPointsIt).Y() * sin( 2. * KConst::Pi() * (*tPartitionIt) * tAngle + aAngleStart* KConst::Pi()/180. );
            tPoint.Z() = (*tPointsIt).X();
            tGroup.push_back( tPoint );
            
        }
        aMesh.fData.push_back( tGroup );
    }

    meshmsg_debug( "rotated open points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

    return;
}
void KGSimpleMesher::ClosedPointsRotatedToTorusMesh( const ClosedPoints& aPoints, const unsigned int& aCount, TorusMesh& aMesh )
{
    aMesh.fData.clear();

    Partition tPartition;
    SymmetricPartition( 0., 1., aCount, 1., tPartition );

    KThreeVector tPoint;
    TubeMesh::Group tGroup;
    for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
    {
        tGroup.clear();
        for( Partition::It tPartitionIt = tPartition.fData.begin(); tPartitionIt != --(tPartition.fData.end()); tPartitionIt++ )
        {
            tPoint.X() = (*tPointsIt).Y() * cos( 2. * KConst::Pi() * (*tPartitionIt) );
            tPoint.Y() = (*tPointsIt).Y() * sin( 2. * KConst::Pi() * (*tPartitionIt) );
            tPoint.Z() = (*tPointsIt).X();
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );
    }

    meshmsg_debug( "rotated closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> torus mesh vertices" << eom );

    return;
}
void KGSimpleMesher::ClosedPointsRotatedToShellMesh( const ClosedPoints& aPoints, const unsigned int& aCount, const double& aPower, ShellMesh& aMesh , const double& aAngleStart, const double& aAngleStop)
{
    aMesh.fData.clear();
    double tAngle = (aAngleStop - aAngleStart)/360;
    Partition tPartition;
    SymmetricPartition( 0., 1., aCount, aPower, tPartition );

    KThreeVector tPoint;
    TubeMesh::Group tGroup;
    for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
    {
        tGroup.clear();
        for( Partition::It tPartitionIt = tPartition.fData.begin(); tPartitionIt != tPartition.fData.end(); tPartitionIt++ )
        {
            tPoint.X() = (*tPointsIt).Y() * cos( 2. * KConst::Pi() * (*tPartitionIt) * tAngle + aAngleStart* KConst::Pi()/180.);
            tPoint.Y() = (*tPointsIt).Y() * sin( 2. * KConst::Pi() * (*tPartitionIt) * tAngle + aAngleStart* KConst::Pi()/180. );
            tPoint.Z() = (*tPointsIt).X();
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );
    }

    meshmsg_debug( "rotated closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> torus mesh vertices" << eom );

    return;
}
void KGSimpleMesher::OpenPointsExtrudedToFlatMesh( const OpenPoints& aPoints, const double& aZMin, const double& aZMax, const unsigned int& aCount, const double& aPower, FlatMesh& aMesh )
{
    aMesh.fData.clear();

    Partition tPartition;
    SymmetricPartition( aZMin, aZMax, aCount, aPower, tPartition );

    KThreeVector tPoint;
    TubeMesh::Group tGroup;
    for( Partition::It tPartitionIt = tPartition.fData.begin(); tPartitionIt != --(tPartition.fData.end()); tPartitionIt++ )
    {
        tGroup.clear();
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = (*tPartitionIt);
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );
    }

    meshmsg_debug( "extruded open points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> flat mesh vertices" << eom );

    return;
}
void KGSimpleMesher::ClosedPointsExtrudedToTubeMesh( const ClosedPoints& aPoints, const double& aZMin, const double& aZMax, const unsigned int& aCount, const double& aPower, TubeMesh& aMesh )
{
    aMesh.fData.clear();

    Partition tPartition;
    SymmetricPartition( aZMin, aZMax, aCount, aPower, tPartition );

    KThreeVector tPoint;
    TubeMesh::Group tGroup;
    for( Partition::It tPartitionIt = tPartition.fData.begin(); tPartitionIt != --(tPartition.fData.end()); tPartitionIt++ )
    {
        tGroup.clear();
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = (*tPartitionIt);
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );
    }

    meshmsg_debug( "extruded closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

    return;
}

    //*********************
    //tesselation functions
    //*********************

void KGSimpleMesher::FlatMeshToTriangles( const FlatMesh& aMesh )
{
    FlatMesh::SetCIt tThisGroup;
    FlatMesh::SetCIt tNextGroup;

    FlatMesh::GroupCIt tThisThisPoint;
    FlatMesh::GroupCIt tThisNextPoint;
    FlatMesh::GroupCIt tNextThisPoint;
    FlatMesh::GroupCIt tNextNextPoint;

        //main hull cells
    tThisGroup = aMesh.fData.begin();
    tNextGroup = ++(aMesh.fData.begin());
    while( tNextGroup != aMesh.fData.end() )
    {
        tThisThisPoint = (*tThisGroup).begin();
        tThisNextPoint = ++((*tThisGroup).begin());
        tNextThisPoint = (*tNextGroup).begin();
        tNextNextPoint = ++((*tNextGroup).begin());
        while( tNextNextPoint != (*tNextGroup).end() )
        {
            Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
            Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

            ++tThisThisPoint;
            ++tThisNextPoint;
            ++tNextThisPoint;
            ++tNextNextPoint;
        }

        ++tThisGroup;
        ++tNextGroup;
    }

    meshmsg_debug( "tesselated flat mesh into <" << fCurrentElements->size() << "> triangles" << eom );

    return;
}
void KGSimpleMesher::ShellMeshToTriangles( const ShellMesh& aMesh )
{   std::cout << "converting shel mesh to triangles" << std::endl;
ShellMesh::SetCIt tThisGroup;
ShellMesh::SetCIt tNextGroup;

ShellMesh::GroupCIt tThisThisPoint;
ShellMesh::GroupCIt tThisNextPoint;
ShellMesh::GroupCIt tNextThisPoint;
ShellMesh::GroupCIt tNextNextPoint;

        //main hull cells
tThisGroup = aMesh.fData.begin();
tNextGroup = ++(aMesh.fData.begin());
while( tNextGroup != aMesh.fData.end() )
{
    tThisThisPoint = (*tThisGroup).begin();
    tThisNextPoint = ++((*tThisGroup).begin());
    tNextThisPoint = (*tNextGroup).begin();
    tNextNextPoint = ++((*tNextGroup).begin());
    while( tNextNextPoint != (*tNextGroup).end() )
    {
        Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
        Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

        ++tThisThisPoint;
        ++tThisNextPoint;
        ++tNextThisPoint;
        ++tNextNextPoint;
    }

    ++tThisGroup;
    ++tNextGroup;
}

meshmsg_debug( "tesselated flat mesh into <" << fCurrentElements->size() << "> triangles" << eom );

return;
}
void KGSimpleMesher::ClosedShellMeshToTriangles( const ShellMesh& aMesh )
{   std::cout << "converting shel mesh to triangles" << std::endl;
ShellMesh::SetCIt tThisGroup;
ShellMesh::SetCIt tNextGroup;

ShellMesh::GroupCIt tThisThisPoint;
ShellMesh::GroupCIt tThisNextPoint;
ShellMesh::GroupCIt tNextThisPoint;
ShellMesh::GroupCIt tNextNextPoint;

        //main hull cells
tThisGroup = aMesh.fData.begin();
tNextGroup = ++(aMesh.fData.begin());
while( tNextGroup != aMesh.fData.end() )
{
    tThisThisPoint = (*tThisGroup).begin();
    tThisNextPoint = ++((*tThisGroup).begin());
    tNextThisPoint = (*tNextGroup).begin();
    tNextNextPoint = ++((*tNextGroup).begin());
    while( tNextNextPoint != (*tNextGroup).end() )
    {
        Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
        Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

        ++tThisThisPoint;
        ++tThisNextPoint;
        ++tNextThisPoint;
        ++tNextNextPoint;
    }

    ++tThisGroup;
    ++tNextGroup;
}

tThisGroup = --(aMesh.fData.end());
    tNextGroup = aMesh.fData.begin();
    tThisThisPoint = (*tThisGroup).begin();
    tThisNextPoint = ++((*tThisGroup).begin());
    tNextThisPoint = (*tNextGroup).begin();
    tNextNextPoint = ++((*tNextGroup).begin());
    while( tNextNextPoint != (*tNextGroup).end() )
    {
        Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
        Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

        ++tThisThisPoint;
        ++tThisNextPoint;
        ++tNextThisPoint;
        ++tNextNextPoint;
    }

    

meshmsg_debug( "tesselated flat mesh into <" << fCurrentElements->size() << "> triangles" << eom );

return;
}
void KGSimpleMesher::TubeMeshToTriangles( const TubeMesh& aMesh )
{
    FlatMesh::SetCIt tThisGroup;
    FlatMesh::SetCIt tNextGroup;

    FlatMesh::GroupCIt tThisThisPoint;
    FlatMesh::GroupCIt tThisNextPoint;
    FlatMesh::GroupCIt tNextThisPoint;
    FlatMesh::GroupCIt tNextNextPoint;

        //main hull cells
    tThisGroup = aMesh.fData.begin();
    tNextGroup = ++(aMesh.fData.begin());
    while( tNextGroup != aMesh.fData.end() )
    {
        tThisThisPoint = (*tThisGroup).begin();
        tThisNextPoint = ++((*tThisGroup).begin());
        tNextThisPoint = (*tNextGroup).begin();
        tNextNextPoint = ++((*tNextGroup).begin());
        while( tNextNextPoint != (*tNextGroup).end() )
        {
            Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
            Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

            ++tThisThisPoint;
            ++tThisNextPoint;
            ++tNextThisPoint;
            ++tNextNextPoint;
        }

        tThisThisPoint = --((*tThisGroup).end());
        tThisNextPoint = (*tThisGroup).begin();
        tNextThisPoint = --((*tNextGroup).end());
        tNextNextPoint = (*tNextGroup).begin();
        Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
        Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

        ++tThisGroup;
        ++tNextGroup;
    }

    meshmsg_debug( "tesselated tube mesh into <" << fCurrentElements->size() << "> triangles" << eom );

    return;
}
void KGSimpleMesher::TubeMeshToTriangles( const KThreeVector& anApexStart, const TubeMesh& aMesh )
{
    TubeMesh::SetCIt tThisGroup;
    TubeMesh::SetCIt tNextGroup;

    TubeMesh::GroupCIt tThisThisPoint;
    TubeMesh::GroupCIt tThisNextPoint;
    TubeMesh::GroupCIt tNextThisPoint;
    TubeMesh::GroupCIt tNextNextPoint;

        //main hull cells
    tThisGroup = aMesh.fData.begin();
    tNextGroup = ++(aMesh.fData.begin());
    while( tNextGroup != aMesh.fData.end() )
    {
        tThisThisPoint = (*tThisGroup).begin();
        tThisNextPoint = ++((*tThisGroup).begin());
        tNextThisPoint = (*tNextGroup).begin();
        tNextNextPoint = ++((*tNextGroup).begin());
        while( tNextNextPoint != (*tNextGroup).end() )
        {
            Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
            Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

            ++tThisThisPoint;
            ++tThisNextPoint;
            ++tNextThisPoint;
            ++tNextNextPoint;
        }

        tThisThisPoint = --((*tThisGroup).end());
        tThisNextPoint = (*tThisGroup).begin();
        tNextThisPoint = --((*tNextGroup).end());
        tNextNextPoint = (*tNextGroup).begin();
        Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
        Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

        ++tThisGroup;
        ++tNextGroup;
    }

        //start cap cells
    tThisGroup = aMesh.fData.begin();
    tThisThisPoint = (*tThisGroup).begin();
    tThisNextPoint = ++((*tThisGroup).begin());
    while( tThisNextPoint != (*tThisGroup).end() )
    {
        Triangle( anApexStart, *tThisThisPoint, *tThisNextPoint );

        ++tThisThisPoint;
        ++tThisNextPoint;
    }

    tThisThisPoint = --((*tThisGroup).end());
    tThisNextPoint = (*tThisGroup).begin();
    Triangle( anApexStart, *tThisThisPoint, *tThisNextPoint );

    meshmsg_debug( "tesselated tube mesh into <" << fCurrentElements->size() << "> triangles" << eom );

    return;
}
void KGSimpleMesher::TubeMeshToTriangles( const TubeMesh& aMesh, const KThreeVector& anApexEnd )
{
    TubeMesh::SetCIt tThisGroup;
    TubeMesh::SetCIt tNextGroup;

    TubeMesh::GroupCIt tThisThisPoint;
    TubeMesh::GroupCIt tThisNextPoint;
    TubeMesh::GroupCIt tNextThisPoint;
    TubeMesh::GroupCIt tNextNextPoint;

        //main hull cells
    tThisGroup = aMesh.fData.begin();
    tNextGroup = ++(aMesh.fData.begin());
    while( tNextGroup != aMesh.fData.end() )
    {
        tThisThisPoint = (*tThisGroup).begin();
        tThisNextPoint = ++((*tThisGroup).begin());
        tNextThisPoint = (*tNextGroup).begin();
        tNextNextPoint = ++((*tNextGroup).begin());
        while( tNextNextPoint != (*tNextGroup).end() )
        {
            Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
            Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

            ++tThisThisPoint;
            ++tThisNextPoint;
            ++tNextThisPoint;
            ++tNextNextPoint;
        }

        tThisThisPoint = --((*tThisGroup).end());
        tThisNextPoint = (*tThisGroup).begin();
        tNextThisPoint = --((*tNextGroup).end());
        tNextNextPoint = (*tNextGroup).begin();
        Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
        Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

        ++tThisGroup;
        ++tNextGroup;
    }

        //end cap cells
    tNextGroup = --(aMesh.fData.end());
    tNextThisPoint = (*tNextGroup).begin();
    tNextNextPoint = ++((*tNextGroup).begin());
    while( tNextNextPoint != (*tNextGroup).end() )
    {
        Triangle( anApexEnd, *tNextThisPoint, *tNextNextPoint );

        ++tNextThisPoint;
        ++tNextNextPoint;
    }

    tNextThisPoint = --((*tNextGroup).end());
    tNextNextPoint = (*tNextGroup).begin();
    Triangle( anApexEnd, *tNextThisPoint, *tNextNextPoint );

    meshmsg_debug( "tesselated tube mesh into <" << fCurrentElements->size() << "> triangles" << eom );

    return;
}
void KGSimpleMesher::TubeMeshToTriangles( const KThreeVector& anApexStart, const TubeMesh& aMesh, const KThreeVector& anApexEnd )
{
    TubeMesh::SetCIt tThisGroup;
    TubeMesh::SetCIt tNextGroup;

    TubeMesh::GroupCIt tThisThisPoint;
    TubeMesh::GroupCIt tThisNextPoint;
    TubeMesh::GroupCIt tNextThisPoint;
    TubeMesh::GroupCIt tNextNextPoint;

        //main hull cells
    tThisGroup = aMesh.fData.begin();
    tNextGroup = ++(aMesh.fData.begin());
    while( tNextGroup != aMesh.fData.end() )
    {
        tThisThisPoint = (*tThisGroup).begin();
        tThisNextPoint = ++((*tThisGroup).begin());
        tNextThisPoint = (*tNextGroup).begin();
        tNextNextPoint = ++((*tNextGroup).begin());
        while( tNextNextPoint != (*tNextGroup).end() )
        {
            Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
            Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

            ++tThisThisPoint;
            ++tThisNextPoint;
            ++tNextThisPoint;
            ++tNextNextPoint;
        }

        tThisThisPoint = --((*tThisGroup).end());
        tThisNextPoint = (*tThisGroup).begin();
        tNextThisPoint = --((*tNextGroup).end());
        tNextNextPoint = (*tNextGroup).begin();
        Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
        Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

        ++tThisGroup;
        ++tNextGroup;
    }

        //start cap cells
    tThisGroup = aMesh.fData.begin();
    tThisThisPoint = (*tThisGroup).begin();
    tThisNextPoint = ++((*tThisGroup).begin());
    while( tThisNextPoint != (*tThisGroup).end() )
    {
        Triangle( anApexStart, *tThisThisPoint, *tThisNextPoint );

        ++tThisThisPoint;
        ++tThisNextPoint;
    }

    tThisThisPoint = --((*tThisGroup).end());
    tThisNextPoint = (*tThisGroup).begin();
    Triangle( anApexStart, *tThisThisPoint, *tThisNextPoint );

        //end cap cells
    tNextGroup = --(aMesh.fData.end());
    tNextThisPoint = (*tNextGroup).begin();
    tNextNextPoint = ++((*tNextGroup).begin());
    while( tNextNextPoint != (*tNextGroup).end() )
    {
        Triangle( anApexEnd, *tNextThisPoint, *tNextNextPoint );

        ++tNextThisPoint;
        ++tNextNextPoint;
    }

    tNextThisPoint = --((*tNextGroup).end());
    tNextNextPoint = (*tNextGroup).begin();
    Triangle( anApexEnd, *tNextThisPoint, *tNextNextPoint );

    meshmsg_debug( "tesselated tube mesh into <" << fCurrentElements->size() << "> triangles" << eom );

    return;
}

void KGSimpleMesher::TorusMeshToTriangles( const TorusMesh& aMesh )
{
    TubeMesh::SetCIt tThisGroup;
    TubeMesh::SetCIt tNextGroup;

    TubeMesh::GroupCIt tThisThisPoint;
    TubeMesh::GroupCIt tThisNextPoint;
    TubeMesh::GroupCIt tNextThisPoint;
    TubeMesh::GroupCIt tNextNextPoint;

        //main hull cells
    tThisGroup = aMesh.fData.begin();
    tNextGroup = ++(aMesh.fData.begin());
    while( tNextGroup != aMesh.fData.end() )
    {
        tThisThisPoint = (*tThisGroup).begin();
        tThisNextPoint = ++((*tThisGroup).begin());
        tNextThisPoint = (*tNextGroup).begin();
        tNextNextPoint = ++((*tNextGroup).begin());
        while( tNextNextPoint != (*tNextGroup).end() )
        {
            Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
            Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

            ++tThisThisPoint;
            ++tThisNextPoint;
            ++tNextThisPoint;
            ++tNextNextPoint;
        }

        tThisThisPoint = --((*tThisGroup).end());
        tThisNextPoint = (*tThisGroup).begin();
        tNextThisPoint = --((*tNextGroup).end());
        tNextNextPoint = (*tNextGroup).begin();
        Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
        Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

        ++tThisGroup;
        ++tNextGroup;
    }

    tThisGroup = --(aMesh.fData.end());
    tNextGroup = aMesh.fData.begin();
    tThisThisPoint = (*tThisGroup).begin();
    tThisNextPoint = ++((*tThisGroup).begin());
    tNextThisPoint = (*tNextGroup).begin();
    tNextNextPoint = ++((*tNextGroup).begin());
    while( tNextNextPoint != (*tNextGroup).end() )
    {
        Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
        Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

        ++tThisThisPoint;
        ++tThisNextPoint;
        ++tNextThisPoint;
        ++tNextNextPoint;
    }

    tThisThisPoint = --((*tThisGroup).end());
    tThisNextPoint = (*tThisGroup).begin();
    tNextThisPoint = --((*tNextGroup).end());
    tNextNextPoint = (*tNextGroup).begin();
    Triangle( *tThisThisPoint, *tThisNextPoint, *tNextThisPoint );
    Triangle( *tNextNextPoint, *tNextThisPoint, *tThisNextPoint );

    meshmsg_debug( "tesselated torus mesh into <" << fCurrentElements->size() << "> triangles" << eom );
}

void KGSimpleMesher::Triangle( const KThreeVector& aFirst, const KThreeVector& aSecond, const KThreeVector& aThird )
{
    fCurrentElements->push_back( new KGMeshTriangle( aFirst, aSecond, aThird ) );
    return;
}

}
