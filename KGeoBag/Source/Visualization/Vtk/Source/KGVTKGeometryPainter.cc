#include "KGVTKGeometryPainter.hh"
#include "KGVisualizationMessage.hh"

#include "KFile.h"
using katrin::KFile;

#include "KConst.h"
using katrin::KConst;

#include "vtkQuad.h"
#include "vtkTriangle.h"
#include "vtkCellArray.h"
#include "vtkPolyDataMapper.h"
#include "vtkAppendPolyData.h"
#include "vtkDepthSortPolyData.h"

#include <cmath>

namespace KGeoBag
{

    KGVTKGeometryPainter::KGVTKGeometryPainter() :
            fFile(),
            fPath( "" ),
            fSurfaces(),
            fSpaces(),
            fDefaultData(),
            fPoints( vtkSmartPointer< vtkPoints >::New() ),
            fCells( vtkSmartPointer< vtkCellArray >::New() ),
            fColors( vtkSmartPointer< vtkUnsignedCharArray >::New() ),
            fPolyData( vtkSmartPointer< vtkPolyData >::New() ),
            fMapper( vtkSmartPointer< vtkPolyDataMapper >::New() ),
            fActor( vtkSmartPointer< vtkActor >::New() ),
            fCurrentSpace( NULL ),
            fCurrentSurface( NULL ),
            fCurrentData( NULL ),
            fCurrentOrigin( KThreeVector::sZero ),
            fCurrentXAxis( KThreeVector::sXUnit ),
            fCurrentYAxis( KThreeVector::sYUnit ),
            fCurrentZAxis( KThreeVector::sZUnit ),
            fIgnore( true )
    {
        fColors->SetNumberOfComponents( 4 );
        fPolyData->SetPoints( fPoints );
        fPolyData->SetPolys( fCells );
        fPolyData->GetCellData()->SetScalars( fColors );
#ifdef VTK6
        fMapper->SetInputData( fPolyData );
#else
        fMapper->SetInput( fPolyData );
#endif
        fMapper->SetScalarModeToUseCellData();
        fActor->SetMapper( fMapper );
        fDefaultData.SetColor( KGRGBAColor( 255, 255, 255, 100 ) );
        fDefaultData.SetArc( 72 );
    }
    KGVTKGeometryPainter::~KGVTKGeometryPainter()
    {
    }

    void KGVTKGeometryPainter::Render()
    {
        KGSurface* tSurface;
        vector< KGSurface* >::iterator tSurfaceIt;
        for( tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++ )
        {
            tSurface = *tSurfaceIt;
            tSurface->AcceptNode( this );
        }

        KGSpace* tSpace;
        vector< KGSpace* >::iterator tSpaceIt;
        for( tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); tSpaceIt++ )
        {
            tSpace = *tSpaceIt;
            tSpace->AcceptNode( this );
        }

        return;
    }
    void KGVTKGeometryPainter::Display()
    {
        if( fDisplayEnabled == true )
        {
            vtkSmartPointer< vtkRenderer > vRenderer = fWindow->GetRenderer();
            vRenderer->AddActor( fActor );
        }
        return;
    }
    void KGVTKGeometryPainter::Write()
    {
        if( fWriteEnabled == true )
        {
            string tFile;

            if( fFile.length() > 0 )
            {
            	if ( fPath.empty() )
            	{
					tFile = string( OUTPUT_DEFAULT_DIR ) + string( "/" ) + fFile;
            	}
            	else
            	{
            		tFile = fPath + string( "/" ) + fFile;
            	}
            }
            else
            {
                tFile = string( OUTPUT_DEFAULT_DIR ) + string( "/" ) + GetName() + string( ".vtp" );
            }

            vismsg( eNormal ) << "vtk geometry painter <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells() << "> cells to file <" << tFile << ">" << eom;

            vtkSmartPointer< vtkXMLPolyDataWriter > vWriter = fWindow->GetWriter();
            vWriter->SetFileName( tFile.c_str() );
            vWriter->SetDataModeToBinary();
#ifdef VTK6
            vWriter->SetInputData( fPolyData );
#else
            vWriter->SetInput( fPolyData );
#endif
            vWriter->Write();
        }
        return;
    }

    void KGVTKGeometryPainter::SetFile( const string& aFile )
    {
        fFile = aFile;
        return;
    }
    const string& KGVTKGeometryPainter::GetFile() const
    {
        return fFile;
    }
    void KGVTKGeometryPainter::SetPath( const string& aPath )
    {
        fPath = aPath;
        return;
    }

    void KGVTKGeometryPainter::AddSurface( KGSurface* aSurface )
    {
        fSurfaces.push_back( aSurface );
        return;
    }
    void KGVTKGeometryPainter::AddSpace( KGSpace* aSpace )
    {
        fSpaces.push_back( aSpace );
        return;
    }

    //****************
    //surface visitors
    //****************

    void KGVTKGeometryPainter::VisitSurface( KGSurface* aSurface )
    {
        fCurrentSurface = aSurface;
        fCurrentOrigin = aSurface->GetOrigin();
        fCurrentXAxis = aSurface->GetXAxis();
        fCurrentYAxis = aSurface->GetYAxis();
        fCurrentZAxis = aSurface->GetZAxis();

        if( aSurface->HasExtension< KGAppearance >() == true )
        {
            fCurrentData = aSurface->AsExtension< KGAppearance >();
        }
        else
        {
            fCurrentData = &fDefaultData;
        }

        if( fCurrentSpace != NULL )
        {
            for( vector< KGSurface* >::const_iterator tIt = fCurrentSpace->GetBoundaries()->begin(); tIt != fCurrentSpace->GetBoundaries()->end(); tIt++ )
            {
                if( (*tIt) == fCurrentSurface )
                {
                    if( fCurrentData == &fDefaultData )
                    {
                        fIgnore = true;
                    }
                    else
                    {
                        fIgnore = false;
                    }
                }
            }
        }
        else
        {
            fIgnore = false;
        }

        return;
    }
    void KGVTKGeometryPainter::VisitFlattenedClosedPathSurface( KGFlattenedCircleSurface* aFlattenedCircleSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create circle points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aFlattenedCircleSurface->Path().operator ->(), tCirclePoints );

        //create flattened points
        KThreeVector tApexPoint;
        TubeMesh tMeshPoints;
        ClosedPointsFlattenedToTubeMeshAndApex( tCirclePoints, aFlattenedCircleSurface->Path()->Centroid(), aFlattenedCircleSurface->Z(), tMeshPoints, tApexPoint );

        //create mesh
        TubeMeshToVTK( tMeshPoints, tApexPoint );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitFlattenedClosedPathSurface( KGFlattenedPolyLoopSurface* aFlattenedPolyLoopSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create circle points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aFlattenedPolyLoopSurface->Path().operator ->(), tPolyLoopPoints );

        //create flattened points
        KThreeVector tApexPoint;
        TubeMesh tMeshPoints;
        ClosedPointsFlattenedToTubeMeshAndApex( tPolyLoopPoints, aFlattenedPolyLoopSurface->Path()->Centroid(), aFlattenedPolyLoopSurface->Z(), tMeshPoints, tApexPoint );

        //create mesh
        TubeMeshToVTK( tMeshPoints, tApexPoint );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitRotatedPathSurface( KGRotatedLineSegmentSurface* aRotatedLineSegmentSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create line segment points
        OpenPoints tLineSegmentPoints;
        LineSegmentToOpenPoints( aRotatedLineSegmentSurface->Path().operator ->(), tLineSegmentPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tLineSegmentPoints, tMeshPoints );

        //surgery
        bool tHasStart = false;
        KThreeVector tStartApex;
        if( aRotatedLineSegmentSurface->Path()->Start().Y() == 0. )
        {
            tHasStart = true;
            tStartApex.SetComponents( 0., 0., aRotatedLineSegmentSurface->Path()->Start().X() );
            tMeshPoints.fData.pop_front();
        }

        bool tHasEnd = false;
        KThreeVector tEndApex;
        if( aRotatedLineSegmentSurface->Path()->End().Y() == 0. )
        {
            tHasEnd = true;
            tEndApex.SetComponents( 0., 0., aRotatedLineSegmentSurface->Path()->End().X() );
            tMeshPoints.fData.pop_back();
        }

        //create mesh
        if( tHasStart == true )
        {
            if( tHasEnd == true )
            {
                TubeMeshToVTK( tStartApex, tMeshPoints, tEndApex );
            }
            else
            {
                TubeMeshToVTK( tStartApex, tMeshPoints );
            }
        }
        else
        {
            if( tHasEnd == true )
            {
                TubeMeshToVTK( tMeshPoints, tEndApex );
            }
            else
            {
                TubeMeshToVTK( tMeshPoints );
            }
        }

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitRotatedPathSurface( KGRotatedArcSegmentSurface* aRotatedArcSegmentSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create arc segment points
        OpenPoints tArcSegmentPoints;
        ArcSegmentToOpenPoints( aRotatedArcSegmentSurface->Path().operator ->(), tArcSegmentPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tArcSegmentPoints, tMeshPoints );

        //surgery
        bool tHasStart = false;
        KThreeVector tStartApex;
        if( aRotatedArcSegmentSurface->Path()->Start().Y() == 0. )
        {
            tHasStart = true;
            tStartApex.SetComponents( 0., 0., aRotatedArcSegmentSurface->Path()->Start().X() );
            tMeshPoints.fData.pop_front();
        }

        bool tHasEnd = false;
        KThreeVector tEndApex;
        if( aRotatedArcSegmentSurface->Path()->End().Y() == 0. )
        {
            tHasEnd = true;
            tEndApex.SetComponents( 0., 0., aRotatedArcSegmentSurface->Path()->End().X() );
            tMeshPoints.fData.pop_back();
        }

        //create mesh
        if( tHasStart == true )
        {
            if( tHasEnd == true )
            {
                TubeMeshToVTK( tStartApex, tMeshPoints, tEndApex );
            }
            else
            {
                TubeMeshToVTK( tStartApex, tMeshPoints );
            }
        }
        else
        {
            if( tHasEnd == true )
            {
                TubeMeshToVTK( tMeshPoints, tEndApex );
            }
            else
            {
                TubeMeshToVTK( tMeshPoints );
            }
        }

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitRotatedPathSurface( KGRotatedPolyLineSurface* aRotatedPolyLineSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly line points
        OpenPoints tPolyLinePoints;
        PolyLineToOpenPoints( aRotatedPolyLineSurface->Path().operator ->(), tPolyLinePoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tPolyLinePoints, tMeshPoints );

        //surgery
        bool tHasStart = false;
        KThreeVector tStartApex;
        if( aRotatedPolyLineSurface->Path()->Start().Y() == 0. )
        {
            tHasStart = true;
            tStartApex.SetComponents( 0., 0., aRotatedPolyLineSurface->Path()->Start().X() );
            tMeshPoints.fData.pop_front();
        }

        bool tHasEnd = false;
        KThreeVector tEndApex;
        if( aRotatedPolyLineSurface->Path()->End().Y() == 0. )
        {
            tHasEnd = true;
            tEndApex.SetComponents( 0., 0., aRotatedPolyLineSurface->Path()->End().X() );
            tMeshPoints.fData.pop_back();
        }

        //create mesh
        if( tHasStart == true )
        {
            if( tHasEnd == true )
            {
                TubeMeshToVTK( tStartApex, tMeshPoints, tEndApex );
            }
            else
            {
                TubeMeshToVTK( tStartApex, tMeshPoints );
            }
        }
        else
        {
            if( tHasEnd == true )
            {
                TubeMeshToVTK( tMeshPoints, tEndApex );
            }
            else
            {
                TubeMeshToVTK( tMeshPoints );
            }
        }

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitRotatedPathSurface( KGRotatedCircleSurface* aRotatedCircleSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly line points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aRotatedCircleSurface->Path().operator ->(), tCirclePoints );

        //create rotated points
        TorusMesh tMeshPoints;
        ClosedPointsRotatedToTorusMesh( tCirclePoints, tMeshPoints );

        //create mesh
        TorusMeshToVTK( tMeshPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitRotatedPathSurface( KGRotatedPolyLoopSurface* aRotatedPolyLoopSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly loop points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aRotatedPolyLoopSurface->Path().operator ->(), tPolyLoopPoints );

        //create rotated points
        TorusMesh tMeshPoints;
        ClosedPointsRotatedToTorusMesh( tPolyLoopPoints, tMeshPoints );

        //create mesh
        TorusMeshToVTK( tMeshPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
void KGVTKGeometryPainter::VisitShellPathSurface( KGShellLineSegmentSurface* aShellLineSegmentSurface )
{ 
    if( fIgnore == true )
    {
        return;
    }

        //create line segment points
    OpenPoints tLineSegmentPoints;
    LineSegmentToOpenPoints( aShellLineSegmentSurface->Path().operator ->(), tLineSegmentPoints );

        //create rotated points
    ShellMesh tMeshPoints;
    OpenPointsRotatedToShellMesh( tLineSegmentPoints, tMeshPoints, aShellLineSegmentSurface->AngleStart(), aShellLineSegmentSurface->AngleStop() );


    ShellMeshToVTK( tMeshPoints );
        //clear surface
    fCurrentSurface = NULL;

    return;
}
void KGVTKGeometryPainter::VisitShellPathSurface( KGShellArcSegmentSurface* aShellArcSegmentSurface )
{ 
    if( fIgnore == true )
    {
        return;
    }

        //create line segment points
    OpenPoints tArcSegmentPoints;
    ArcSegmentToOpenPoints( aShellArcSegmentSurface->Path().operator ->(), tArcSegmentPoints );

        //create rotated points
    ShellMesh tMeshPoints;
    OpenPointsRotatedToShellMesh( tArcSegmentPoints, tMeshPoints, aShellArcSegmentSurface->AngleStart(), aShellArcSegmentSurface->AngleStop() );

    
    ShellMeshToVTK( tMeshPoints );
        //clear surface
    fCurrentSurface = NULL;

    return;
}
void KGVTKGeometryPainter::VisitShellPathSurface( KGShellPolyLineSurface* aShellPolyLineSurface )
{ std::cout << "visiting Shel Poly Line Surface" << std::endl;
    if( fIgnore == true )
    {
        return;
    }

        //create poly line points
    OpenPoints tPolyLinePoints;
    PolyLineToOpenPoints( aShellPolyLineSurface->Path().operator ->(), tPolyLinePoints );

        //create shell points
    ShellMesh tMeshPoints;
    OpenPointsRotatedToShellMesh( tPolyLinePoints, tMeshPoints, aShellPolyLineSurface->AngleStart(), aShellPolyLineSurface->AngleStop() );

    
        //create mesh
    
    ShellMeshToVTK( tMeshPoints );
    

        //clear surface
    fCurrentSurface = NULL;

    return;
}
void KGVTKGeometryPainter::VisitShellPathSurface( KGShellPolyLoopSurface* aShellPolyLoopSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly loop points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aShellPolyLoopSurface->Path().operator ->(), tPolyLoopPoints );

        //create rotated points
        ShellMesh tMeshPoints;
        ClosedPointsRotatedToShellMesh( tPolyLoopPoints, tMeshPoints, aShellPolyLoopSurface->AngleStart(), aShellPolyLoopSurface->AngleStop()  );

        //create mesh
        ClosedShellMeshToVTK( tMeshPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitShellPathSurface( KGShellCircleSurface* aShellCircleSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly line points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aShellCircleSurface->Path().operator ->(), tCirclePoints );

        //create rotated points
        ShellMesh tMeshPoints;
        ClosedPointsRotatedToShellMesh( tCirclePoints, tMeshPoints , aShellCircleSurface->AngleStart(), aShellCircleSurface->AngleStop() );

        //create mesh
        ClosedShellMeshToVTK( tMeshPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitExtrudedPathSurface( KGExtrudedLineSegmentSurface* aExtrudedLineSegmentSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create line segment points
        OpenPoints tLineSegmentPoints;
        LineSegmentToOpenPoints( aExtrudedLineSegmentSurface->Path().operator ->(), tLineSegmentPoints );

        //create extruded points
        FlatMesh tMeshPoints;
        OpenPointsExtrudedToFlatMesh( tLineSegmentPoints, aExtrudedLineSegmentSurface->ZMin(), aExtrudedLineSegmentSurface->ZMax(), tMeshPoints );

        //create mesh
        FlatMeshToVTK( tMeshPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitExtrudedPathSurface( KGExtrudedArcSegmentSurface* aExtrudedArcSegmentSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create arc segment points
        OpenPoints tArcSegmentPoints;
        ArcSegmentToOpenPoints( aExtrudedArcSegmentSurface->Path().operator ->(), tArcSegmentPoints );

        //create extruded points
        FlatMesh tMeshPoints;
        OpenPointsExtrudedToFlatMesh( tArcSegmentPoints, aExtrudedArcSegmentSurface->ZMin(), aExtrudedArcSegmentSurface->ZMax(), tMeshPoints );

        //create mesh
        FlatMeshToVTK( tMeshPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitExtrudedPathSurface( KGExtrudedPolyLineSurface* aExtrudedPolyLineSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly line points
        OpenPoints tPolyLinePoints;
        PolyLineToOpenPoints( aExtrudedPolyLineSurface->Path().operator ->(), tPolyLinePoints );

        //create extruded points
        FlatMesh tMeshPoints;
        OpenPointsExtrudedToFlatMesh( tPolyLinePoints, aExtrudedPolyLineSurface->ZMin(), aExtrudedPolyLineSurface->ZMax(), tMeshPoints );

        //create mesh
        FlatMeshToVTK( tMeshPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitExtrudedPathSurface( KGExtrudedCircleSurface* aExtrudedCircleSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly line points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aExtrudedCircleSurface->Path().operator ->(), tCirclePoints );

        //create rotated points
        TubeMesh tMeshPoints;
        ClosedPointsExtrudedToTubeMesh( tCirclePoints, aExtrudedCircleSurface->ZMin(), aExtrudedCircleSurface->ZMax(), tMeshPoints );

        //create mesh
        TubeMeshToVTK( tMeshPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitExtrudedPathSurface( KGExtrudedPolyLoopSurface* aExtrudedPolyLoopSurface )
    {
        if( fIgnore == true )
        {
            return;
        }

        //create poly loop points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aExtrudedPolyLoopSurface->Path().operator ->(), tPolyLoopPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        ClosedPointsExtrudedToTubeMesh( tPolyLoopPoints, aExtrudedPolyLoopSurface->ZMin(), aExtrudedPolyLoopSurface->ZMax(), tMeshPoints );

        //create mesh
        TubeMeshToVTK( tMeshPoints );

        //clear surface
        fCurrentSurface = NULL;

        return;
    }

    //**************
    //space visitors
    //**************

    void KGVTKGeometryPainter::VisitSpace( KGSpace* aSpace )
    {
        fCurrentSpace = aSpace;
        fCurrentOrigin = aSpace->GetOrigin();
        fCurrentXAxis = aSpace->GetXAxis();
        fCurrentYAxis = aSpace->GetYAxis();
        fCurrentZAxis = aSpace->GetZAxis();

        if( aSpace->HasExtension< KGAppearance >() == true )
        {
            fCurrentData = aSpace->AsExtension< KGAppearance >();
        }
        else
        {
            fCurrentData = &fDefaultData;
        }

        return;
    }
    void KGVTKGeometryPainter::VisitRotatedOpenPathSpace( KGRotatedLineSegmentSpace* aRotatedLineSegmentSpace )
    {
        //create line segment points
        OpenPoints tLineSegmentPoints;
        LineSegmentToOpenPoints( aRotatedLineSegmentSpace->Path().operator ->(), tLineSegmentPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tLineSegmentPoints, tMeshPoints );

        //create start circle points
        ClosedPoints tStartCirclePoints;
        CircleToClosedPoints( aRotatedLineSegmentSpace->StartPath().operator ->(), tStartCirclePoints );

        //create start flattened mesh points
        TubeMesh tStartMeshPoints;
        KThreeVector tStartApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tStartCirclePoints, aRotatedLineSegmentSpace->StartPath()->Centroid(), aRotatedLineSegmentSpace->Path()->Start().X(), tStartMeshPoints, tStartApex );

        //create end circle points
        ClosedPoints tEndCirclePoints;
        CircleToClosedPoints( aRotatedLineSegmentSpace->EndPath().operator ->(), tEndCirclePoints );

        //create end flattened mesh points
        TubeMesh tEndMeshPoints;
        KThreeVector tEndApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tEndCirclePoints, aRotatedLineSegmentSpace->EndPath()->Centroid(), aRotatedLineSegmentSpace->Path()->End().X(), tEndMeshPoints, tEndApex );

        //surgery
        if( aRotatedLineSegmentSpace->Path()->Start().Y() > 0 )
        {
            TubeMesh::SetIt tCircleIt = ++(tStartMeshPoints.fData.begin());
            while( tCircleIt != tStartMeshPoints.fData.end() )
            {
                tMeshPoints.fData.push_front( *tCircleIt );
                ++tCircleIt;
            }
        }
        else
        {
            tMeshPoints.fData.pop_front();
        }

        if( aRotatedLineSegmentSpace->Path()->End().Y() > 0 )
        {
            TubeMesh::SetIt tCircleIt = ++(tEndMeshPoints.fData.begin());
            while( tCircleIt != tEndMeshPoints.fData.end() )
            {
                tMeshPoints.fData.push_back( *tCircleIt );
                ++tCircleIt;
            }
        }
        else
        {
            tMeshPoints.fData.pop_back();
        }

        TubeMeshToVTK( tStartApex, tMeshPoints, tEndApex );

        //clear space
        fCurrentSpace = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitRotatedOpenPathSpace( KGRotatedArcSegmentSpace* aRotatedArcSegmentSpace )
    {
        //create line segment points
        OpenPoints tArcSegmentPoints;
        ArcSegmentToOpenPoints( aRotatedArcSegmentSpace->Path().operator ->(), tArcSegmentPoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tArcSegmentPoints, tMeshPoints );

        //create start circle points
        ClosedPoints tStartCirclePoints;
        CircleToClosedPoints( aRotatedArcSegmentSpace->StartPath().operator ->(), tStartCirclePoints );

        //create start flattened mesh points
        TubeMesh tStartMeshPoints;
        KThreeVector tStartApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tStartCirclePoints, aRotatedArcSegmentSpace->StartPath()->Centroid(), aRotatedArcSegmentSpace->Path()->Start().X(), tStartMeshPoints, tStartApex );

        //create end circle points
        ClosedPoints tEndCirclePoints;
        CircleToClosedPoints( aRotatedArcSegmentSpace->EndPath().operator ->(), tEndCirclePoints );

        //create end flattened mesh points
        TubeMesh tEndMeshPoints;
        KThreeVector tEndApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tEndCirclePoints, aRotatedArcSegmentSpace->EndPath()->Centroid(), aRotatedArcSegmentSpace->Path()->End().X(), tEndMeshPoints, tEndApex );

        //surgery
        if( aRotatedArcSegmentSpace->Path()->Start().Y() > 0 )
        {
            TubeMesh::SetIt tCircleIt = ++(tStartMeshPoints.fData.begin());
            while( tCircleIt != tStartMeshPoints.fData.end() )
            {
                tMeshPoints.fData.push_front( *tCircleIt );
                ++tCircleIt;
            }
        }
        else
        {
            tMeshPoints.fData.pop_front();
        }

        if( aRotatedArcSegmentSpace->Path()->End().Y() > 0 )
        {
            TubeMesh::SetIt tCircleIt = ++(tEndMeshPoints.fData.begin());
            while( tCircleIt != tEndMeshPoints.fData.end() )
            {
                tMeshPoints.fData.push_back( *tCircleIt );
                ++tCircleIt;
            }
        }
        else
        {
            tMeshPoints.fData.pop_back();
        }

        TubeMeshToVTK( tStartApex, tMeshPoints, tEndApex );

        //clear space
        fCurrentSpace = NULL;

        return;
    }

    void KGVTKGeometryPainter::VisitRotatedOpenPathSpace( KGRotatedPolyLineSpace* aRotatedPolyLineSpace )
    {
        //create line segment points
        OpenPoints tPolyLinePoints;
        PolyLineToOpenPoints( aRotatedPolyLineSpace->Path().operator ->(), tPolyLinePoints );

        //create rotated points
        TubeMesh tMeshPoints;
        OpenPointsRotatedToTubeMesh( tPolyLinePoints, tMeshPoints );

        //create start circle points
        ClosedPoints tStartCirclePoints;
        CircleToClosedPoints( aRotatedPolyLineSpace->StartPath().operator ->(), tStartCirclePoints );

        //create start flattened mesh points
        TubeMesh tStartMeshPoints;
        KThreeVector tStartApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tStartCirclePoints, aRotatedPolyLineSpace->StartPath()->Centroid(), aRotatedPolyLineSpace->Path()->Start().X(), tStartMeshPoints, tStartApex );

        //create end circle points
        ClosedPoints tEndCirclePoints;
        CircleToClosedPoints( aRotatedPolyLineSpace->EndPath().operator ->(), tEndCirclePoints );

        //create end flattened mesh points
        TubeMesh tEndMeshPoints;
        KThreeVector tEndApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tEndCirclePoints, aRotatedPolyLineSpace->EndPath()->Centroid(), aRotatedPolyLineSpace->Path()->End().X(), tEndMeshPoints, tEndApex );

        //surgery
        if( aRotatedPolyLineSpace->Path()->Start().Y() > 0 )
        {
            TubeMesh::SetIt tCircleIt = ++(tStartMeshPoints.fData.begin());
            while( tCircleIt != tStartMeshPoints.fData.end() )
            {
                tMeshPoints.fData.push_front( *tCircleIt );
                ++tCircleIt;
            }
        }
        else
        {
            tMeshPoints.fData.pop_front();
        }

        if( aRotatedPolyLineSpace->Path()->End().Y() > 0 )
        {
            TubeMesh::SetIt tCircleIt = ++(tEndMeshPoints.fData.begin());
            while( tCircleIt != tEndMeshPoints.fData.end() )
            {
                tMeshPoints.fData.push_back( *tCircleIt );
                ++tCircleIt;
            }
        }
        else
        {
            tMeshPoints.fData.pop_back();
        }

        TubeMeshToVTK( tStartApex, tMeshPoints, tEndApex );

        //clear space
        fCurrentSpace = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitRotatedClosedPathSpace( KGRotatedCircleSpace* aRotatedCircleSpace )
    {
        //create poly line points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aRotatedCircleSpace->Path().operator ->(), tCirclePoints );

        //create rotated points
        TorusMesh tMeshPoints;
        ClosedPointsRotatedToTorusMesh( tCirclePoints, tMeshPoints );

        //create mesh
        TorusMeshToVTK( tMeshPoints );

        return;
    }
    void KGVTKGeometryPainter::VisitRotatedClosedPathSpace( KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace )
    {
        //create poly line points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aRotatedPolyLoopSpace->Path().operator ->(), tPolyLoopPoints );

        //create rotated points
        TorusMesh tMeshPoints;
        ClosedPointsRotatedToTorusMesh( tPolyLoopPoints, tMeshPoints );

        //create mesh
        TorusMeshToVTK( tMeshPoints );

        //clear space
        fCurrentSpace = NULL;

        return;
    }
    void KGVTKGeometryPainter::VisitExtrudedClosedPathSpace( KGExtrudedCircleSpace* aExtrudedCircleSpace )
    {
        //create circle points
        ClosedPoints tCirclePoints;
        CircleToClosedPoints( aExtrudedCircleSpace->Path().operator->(), tCirclePoints );

        //create extruded points
        TubeMesh tMeshPoints;
        ClosedPointsExtrudedToTubeMesh( tCirclePoints, aExtrudedCircleSpace->ZMin(), aExtrudedCircleSpace->ZMax(), tMeshPoints );

        //create start flattened mesh points
        TubeMesh tStartMeshPoints;
        KThreeVector tStartApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tCirclePoints, aExtrudedCircleSpace->Path()->Centroid(), aExtrudedCircleSpace->ZMin(), tStartMeshPoints, tStartApex );

        //create end flattened mesh points
        TubeMesh tEndMeshPoints;
        KThreeVector tEndApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tCirclePoints, aExtrudedCircleSpace->Path()->Centroid(), aExtrudedCircleSpace->ZMax(), tEndMeshPoints, tEndApex );

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
        TubeMeshToVTK( tStartApex, tMeshPoints, tEndApex );

        //clear space
        fCurrentSpace = NULL;

        return;
    }

    void KGVTKGeometryPainter::VisitExtrudedClosedPathSpace( KGExtrudedPolyLoopSpace* aExtrudedPolyLoopSpace )
    {
        //create circle points
        ClosedPoints tPolyLoopPoints;
        PolyLoopToClosedPoints( aExtrudedPolyLoopSpace->Path().operator->(), tPolyLoopPoints );

        //create extruded points
        TubeMesh tMeshPoints;
        ClosedPointsExtrudedToTubeMesh( tPolyLoopPoints, aExtrudedPolyLoopSpace->ZMin(), aExtrudedPolyLoopSpace->ZMax(), tMeshPoints );

        //create start flattened mesh points
        TubeMesh tStartMeshPoints;
        KThreeVector tStartApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tPolyLoopPoints, aExtrudedPolyLoopSpace->Path()->Centroid(), aExtrudedPolyLoopSpace->ZMin(), tStartMeshPoints, tStartApex );

        //create end flattened mesh points
        TubeMesh tEndMeshPoints;
        KThreeVector tEndApex;
        ClosedPointsFlattenedToTubeMeshAndApex( tPolyLoopPoints, aExtrudedPolyLoopSpace->Path()->Centroid(), aExtrudedPolyLoopSpace->ZMax(), tEndMeshPoints, tEndApex );

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
        TubeMeshToVTK( tStartApex, tMeshPoints, tEndApex );

        //clear space
        fCurrentSpace = NULL;

        return;
    }

    void KGVTKGeometryPainter::LocalToGlobal( const KThreeVector& aLocal, KThreeVector& aGlobal )
    {
        aGlobal = fCurrentOrigin + aLocal.X() * fCurrentXAxis + aLocal.Y() * fCurrentYAxis + aLocal.Z() * fCurrentZAxis;
        return;
    }

    //****************
    //points functions
    //****************

    void KGVTKGeometryPainter::LineSegmentToOpenPoints( const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints )
    {
        aPoints.fData.clear();

        aPoints.fData.push_back( aLineSegment->At( 0. ) );
        aPoints.fData.push_back( aLineSegment->At( aLineSegment->Length() ) );

        vismsg_debug( "line segment partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }
    void KGVTKGeometryPainter::ArcSegmentToOpenPoints( const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints )
    {
        aPoints.fData.clear();

        double tArcFraction = anArcSegment->Length() / (2. * KConst::Pi() * anArcSegment->Radius());
        unsigned int tArc = (unsigned int) (ceil( tArcFraction * (double) (fCurrentData->GetArc()) ));

        double tFraction;
        unsigned int tCount;
        for( tCount = 0; tCount <= tArc; tCount++ )
        {
            tFraction = anArcSegment->Length() * ((double) (tCount) / (double) (tArc));
            aPoints.fData.push_back( anArcSegment->At( tFraction ) );
        }

        vismsg_debug( "arc segment partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }
    void KGVTKGeometryPainter::PolyLineToOpenPoints( const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints )
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

        vismsg_debug( "poly line partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }
    void KGVTKGeometryPainter::CircleToClosedPoints( const KGPlanarCircle* aCircle, ClosedPoints& aPoints )
    {
        aPoints.fData.clear();

        unsigned int tArc = fCurrentData->GetArc();

        double tFraction;
        unsigned int tCount;
        for( tCount = 0; tCount < tArc; tCount++ )
        {
            tFraction = aCircle->Length() * ((double) (tCount) / (double) (tArc));
            aPoints.fData.push_back( aCircle->At( tFraction ) );
        }

        vismsg_debug( "circle partitioned into <" << aPoints.fData.size() << "> closed points vertices" << eom );

        return;
    }
    void KGVTKGeometryPainter::PolyLoopToClosedPoints( const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints )
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
            tSubPoints.fData.clear();

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

        vismsg_debug( "poly loop partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom );

        return;
    }

    //**************
    //mesh functions
    //**************

    void KGVTKGeometryPainter::ClosedPointsFlattenedToTubeMeshAndApex( const ClosedPoints& aPoints, const KTwoVector& aCentroid, const double& aZ, TubeMesh& aMesh, KThreeVector& anApex )
    {
        KThreeVector tPoint;
        TubeMesh::Group tGroup;
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = aZ;
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );
        anApex.X() = aCentroid.X();
        anApex.Y() = aCentroid.Y();
        anApex.Z() = aZ;

        vismsg_debug( "flattened closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

        return;
    }
    void KGVTKGeometryPainter::OpenPointsRotatedToTubeMesh( const OpenPoints& aPoints, TubeMesh& aMesh )
    {
        unsigned int tArc = fCurrentData->GetArc();

        double tFraction;
        unsigned int tCount;

        KThreeVector tPoint;
        TubeMesh::Group tGroup;
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tGroup.clear();
            for( tCount = 0; tCount < tArc; tCount++ )
            {
                tFraction = (double) (tCount) / (double) (tArc);

                tPoint.X() = (*tPointsIt).Y() * cos( 2. * KConst::Pi() * tFraction );
                tPoint.Y() = (*tPointsIt).Y() * sin( 2. * KConst::Pi() * tFraction );
                tPoint.Z() = (*tPointsIt).X();
                tGroup.push_back( tPoint );
            }
            aMesh.fData.push_back( tGroup );
        }

        vismsg_debug( "rotated open points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

        return;
    }
void KGVTKGeometryPainter::OpenPointsRotatedToShellMesh( const OpenPoints& aPoints, ShellMesh& aMesh , const double& aAngleStart, const double& aAngleStop)
{   std::cout << "rotating open points to shell mesh" << std::endl;
    unsigned int tArc = fCurrentData->GetArc();

    double tFraction;
    unsigned int tCount;
    double tAngle = (aAngleStop - aAngleStart)/360;

    KThreeVector tPoint;
    ShellMesh::Group tGroup;
    for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
    {
        tGroup.clear();
        for( tCount = 0; tCount <= tArc; tCount++ )
        {
            tFraction = (double) (tCount) / (double) (tArc);

            tPoint.X() = (*tPointsIt).Y() * cos( 2. * KConst::Pi() * tFraction * tAngle + aAngleStart* KConst::Pi()/180.);
            tPoint.Y() = (*tPointsIt).Y() * sin(  2. * KConst::Pi() * tFraction * tAngle + aAngleStart* KConst::Pi()/180. );
            tPoint.Z() = (*tPointsIt).X();
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );
    }


    vismsg_debug( "rotated open points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

    return;
}

void KGVTKGeometryPainter::ClosedPointsRotatedToTorusMesh( const ClosedPoints& aPoints, TorusMesh& aMesh )
    {
        unsigned int tArc = fCurrentData->GetArc();

        double tFraction;
        unsigned int tCount;

        KThreeVector tPoint;
        TubeMesh::Group tGroup;
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tGroup.clear();
            for( tCount = 0; tCount < tArc; tCount++ )
            {
                tFraction = (double) (tCount) / (double) (tArc);

                tPoint.X() = (*tPointsIt).Y() * cos( 2. * KConst::Pi() * tFraction );
                tPoint.Y() = (*tPointsIt).Y() * sin( 2. * KConst::Pi() * tFraction );
                tPoint.Z() = (*tPointsIt).X();
                tGroup.push_back( tPoint );
            }
            aMesh.fData.push_back( tGroup );
        }

        vismsg_debug( "rotated closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> torus mesh vertices" << eom );

        return;
    }
    void KGVTKGeometryPainter::ClosedPointsRotatedToShellMesh( const ClosedPoints& aPoints, ShellMesh& aMesh , const double& aAngleStart, const double& aAngleStop)
    {
        unsigned int tArc = fCurrentData->GetArc();

        double tFraction;
        unsigned int tCount;
        double tAngle = (aAngleStop - aAngleStart)/360;
        KThreeVector tPoint;
        TubeMesh::Group tGroup;
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tGroup.clear();
            for( tCount = 0; tCount <= tArc; tCount++ )
            {
                tFraction = (double) (tCount) / (double) (tArc);

                tPoint.X() = (*tPointsIt).Y() * cos( 2. * KConst::Pi() * tFraction * tAngle + aAngleStart* KConst::Pi()/180.);
            tPoint.Y() = (*tPointsIt).Y() * sin(  2. * KConst::Pi() * tFraction * tAngle + aAngleStart* KConst::Pi()/180. );
            tPoint.Z() = (*tPointsIt).X();
            tGroup.push_back( tPoint );
            }
            aMesh.fData.push_back( tGroup );
        }

        vismsg_debug( "rotated closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> torus mesh vertices" << eom );

        return;
    }
    void KGVTKGeometryPainter::OpenPointsExtrudedToFlatMesh( const OpenPoints& aPoints, const double& aZMin, const double& aZMax, FlatMesh& aMesh )
    {
        KThreeVector tPoint;
        TubeMesh::Group tGroup;

        tGroup.clear();
        for( OpenPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = aZMin;
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );

        tGroup.clear();
        for( OpenPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = aZMax;
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );

        vismsg_debug( "extruded open points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> flat mesh vertices" << eom );

        return;
    }
    void KGVTKGeometryPainter::ClosedPointsExtrudedToTubeMesh( const ClosedPoints& aPoints, const double& aZMin, const double& aZMax, TubeMesh& aMesh )
    {
        KThreeVector tPoint;
        TubeMesh::Group tGroup;

        tGroup.clear();
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = aZMin;
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );

        tGroup.clear();
        for( ClosedPoints::CIt tPointsIt = aPoints.fData.begin(); tPointsIt != aPoints.fData.end(); tPointsIt++ )
        {
            tPoint.X() = (*tPointsIt).X();
            tPoint.Y() = (*tPointsIt).Y();
            tPoint.Z() = aZMax;
            tGroup.push_back( tPoint );
        }
        aMesh.fData.push_back( tGroup );

        vismsg_debug( "extruded closed points into <" << aMesh.fData.size() * aMesh.fData.front().size() << "> tube mesh vertices" << eom );

        return;
    }

    //*******************
    //rendering functions
    //*******************

    void KGVTKGeometryPainter::FlatMeshToVTK( const FlatMesh& aMesh )
    {
        //object allocation
        KThreeVector tPoint;

        deque< vtkIdType > vMeshIdGroup;
        deque< deque< vtkIdType > > vMeshIdSet;

        deque< deque< vtkIdType > >::iterator vThisGroup;
        deque< deque< vtkIdType > >::iterator vNextGroup;

        deque< vtkIdType >::iterator vThisThisPoint;
        deque< vtkIdType >::iterator vThisNextPoint;
        deque< vtkIdType >::iterator vNextThisPoint;
        deque< vtkIdType >::iterator vNextNextPoint;

        vtkSmartPointer< vtkQuad > vQuad;

        //create mesh point ids
        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
        {
            vMeshIdGroup.clear();
            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
            {
                LocalToGlobal( *tGroupIt, tPoint );
                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
            }
            vMeshIdSet.push_back( vMeshIdGroup );
        }

        //create hull cells
        vThisGroup = vMeshIdSet.begin();
        vNextGroup = ++(vMeshIdSet.begin());
        while( vNextGroup != vMeshIdSet.end() )
        {
            vThisThisPoint = vThisGroup->begin();
            vThisNextPoint = ++(vThisGroup->begin());
            vNextThisPoint = vNextGroup->begin();
            vNextNextPoint = ++(vNextGroup->begin());

            while( vNextNextPoint != vNextGroup->end() )
            {
                vQuad = vtkSmartPointer< vtkQuad >::New();
                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
                fCells->InsertNextCell( vQuad );
                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
                ++vThisThisPoint;
                ++vThisNextPoint;
                ++vNextThisPoint;
                ++vNextNextPoint;
            }

            ++vThisGroup;
            ++vNextGroup;
        }

        return;
    }
    void KGVTKGeometryPainter::TubeMeshToVTK( const TubeMesh& aMesh )
    {
        //object allocation
        KThreeVector tPoint;

        deque< vtkIdType > vMeshIdGroup;
        deque< deque< vtkIdType > > vMeshIdSet;

        deque< deque< vtkIdType > >::iterator vThisGroup;
        deque< deque< vtkIdType > >::iterator vNextGroup;

        deque< vtkIdType >::iterator vThisThisPoint;
        deque< vtkIdType >::iterator vThisNextPoint;
        deque< vtkIdType >::iterator vNextThisPoint;
        deque< vtkIdType >::iterator vNextNextPoint;

        vtkSmartPointer< vtkQuad > vQuad;

        //create mesh point ids
        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
        {
            vMeshIdGroup.clear();
            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
            {
                LocalToGlobal( *tGroupIt, tPoint );
                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
            }
            vMeshIdGroup.push_back( vMeshIdGroup.front() );
            vMeshIdSet.push_back( vMeshIdGroup );
        }

        //create hull cells
        vThisGroup = vMeshIdSet.begin();
        vNextGroup = ++(vMeshIdSet.begin());
        while( vNextGroup != vMeshIdSet.end() )
        {
            vThisThisPoint = vThisGroup->begin();
            vThisNextPoint = ++(vThisGroup->begin());
            vNextThisPoint = vNextGroup->begin();
            vNextNextPoint = ++(vNextGroup->begin());

            while( vNextNextPoint != vNextGroup->end() )
            {
                vQuad = vtkSmartPointer< vtkQuad >::New();
                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
                fCells->InsertNextCell( vQuad );
                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
                ++vThisThisPoint;
                ++vThisNextPoint;
                ++vNextThisPoint;
                ++vNextNextPoint;
            }

            ++vThisGroup;
            ++vNextGroup;
        }

        return;
    }
    void KGVTKGeometryPainter::TubeMeshToVTK( const KThreeVector& anApexStart, const TubeMesh& aMesh )
    {
        //object allocation
        KThreeVector tPoint;

        vtkIdType vMeshIdApexStart;
        deque< vtkIdType > vMeshIdGroup;
        deque< deque< vtkIdType > > vMeshIdSet;

        deque< deque< vtkIdType > >::iterator vThisGroup;
        deque< deque< vtkIdType > >::iterator vNextGroup;

        deque< vtkIdType >::iterator vThisThisPoint;
        deque< vtkIdType >::iterator vThisNextPoint;
        deque< vtkIdType >::iterator vNextThisPoint;
        deque< vtkIdType >::iterator vNextNextPoint;

        vtkSmartPointer< vtkTriangle > vTriangle;
        vtkSmartPointer< vtkQuad > vQuad;

        //create apex start point id
        LocalToGlobal( anApexStart, tPoint );
        vMeshIdApexStart = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );

        //create mesh point ids
        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
        {
            vMeshIdGroup.clear();
            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
            {
                LocalToGlobal( *tGroupIt, tPoint );
                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
            }
            vMeshIdGroup.push_back( vMeshIdGroup.front() );
            vMeshIdSet.push_back( vMeshIdGroup );
        }

        //create start cap cells
        vThisThisPoint = vMeshIdSet.front().begin();
        vThisNextPoint = ++(vMeshIdSet.front().begin());
        while( vThisNextPoint != vMeshIdSet.front().end() )
        {
            vTriangle = vtkSmartPointer< vtkTriangle >::New();
            vTriangle->GetPointIds()->SetId( 0, *vThisThisPoint );
            vTriangle->GetPointIds()->SetId( 1, *vThisNextPoint );
            vTriangle->GetPointIds()->SetId( 2, vMeshIdApexStart );
            fCells->InsertNextCell( vTriangle );
            fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
            ++vThisThisPoint;
            ++vThisNextPoint;
        }

        //create hull cells
        vThisGroup = vMeshIdSet.begin();
        vNextGroup = ++(vMeshIdSet.begin());
        while( vNextGroup != vMeshIdSet.end() )
        {
            vThisThisPoint = vThisGroup->begin();
            vThisNextPoint = ++(vThisGroup->begin());
            vNextThisPoint = vNextGroup->begin();
            vNextNextPoint = ++(vNextGroup->begin());

            while( vNextNextPoint != vNextGroup->end() )
            {
                vQuad = vtkSmartPointer< vtkQuad >::New();
                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
                fCells->InsertNextCell( vQuad );
                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
                ++vThisThisPoint;
                ++vThisNextPoint;
                ++vNextThisPoint;
                ++vNextNextPoint;
            }

            ++vThisGroup;
            ++vNextGroup;
        }

        return;
    }
    void KGVTKGeometryPainter::TubeMeshToVTK( const TubeMesh& aMesh, const KThreeVector& anApexEnd )
    {
        //object allocation
        KThreeVector tPoint;

        vtkIdType vMeshIdApexEnd;
        deque< vtkIdType > vMeshIdGroup;
        deque< deque< vtkIdType > > vMeshIdSet;

        deque< deque< vtkIdType > >::iterator vThisGroup;
        deque< deque< vtkIdType > >::iterator vNextGroup;

        deque< vtkIdType >::iterator vThisThisPoint;
        deque< vtkIdType >::iterator vThisNextPoint;
        deque< vtkIdType >::iterator vNextThisPoint;
        deque< vtkIdType >::iterator vNextNextPoint;

        vtkSmartPointer< vtkTriangle > vTriangle;
        vtkSmartPointer< vtkQuad > vQuad;

        //create apex end point id
        LocalToGlobal( anApexEnd, tPoint );
        vMeshIdApexEnd = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );

        //create mesh point ids
        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
        {
            vMeshIdGroup.clear();
            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
            {
                LocalToGlobal( *tGroupIt, tPoint );
                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
            }
            vMeshIdGroup.push_back( vMeshIdGroup.front() );
            vMeshIdSet.push_back( vMeshIdGroup );
        }

        //create hull cells
        vThisGroup = vMeshIdSet.begin();
        vNextGroup = ++(vMeshIdSet.begin());
        while( vNextGroup != vMeshIdSet.end() )
        {
            vThisThisPoint = vThisGroup->begin();
            vThisNextPoint = ++(vThisGroup->begin());
            vNextThisPoint = vNextGroup->begin();
            vNextNextPoint = ++(vNextGroup->begin());

            while( vNextNextPoint != vNextGroup->end() )
            {
                vQuad = vtkSmartPointer< vtkQuad >::New();
                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
                fCells->InsertNextCell( vQuad );
                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
                ++vThisThisPoint;
                ++vThisNextPoint;
                ++vNextThisPoint;
                ++vNextNextPoint;
            }

            ++vThisGroup;
            ++vNextGroup;
        }

        //create end cap cells
        vThisThisPoint = vMeshIdSet.back().begin();
        vThisNextPoint = ++(vMeshIdSet.back().begin());
        while( vThisNextPoint != vMeshIdSet.back().end() )
        {
            vTriangle = vtkSmartPointer< vtkTriangle >::New();
            vTriangle->GetPointIds()->SetId( 0, *vThisThisPoint );
            vTriangle->GetPointIds()->SetId( 1, *vThisNextPoint );
            vTriangle->GetPointIds()->SetId( 2, vMeshIdApexEnd );
            fCells->InsertNextCell( vTriangle );
            fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
            ++vThisThisPoint;
            ++vThisNextPoint;
        }

        return;
    }
    void KGVTKGeometryPainter::TubeMeshToVTK( const KThreeVector& anApexStart, const TubeMesh& aMesh, const KThreeVector& anApexEnd )
    {
    	if ( aMesh.fData.size() == 0 )
    	{
    		vismsg( eWarning ) <<"mesh has size of zero, check your geometry"<<eom;
    	}

        //object allocation
        KThreeVector tPoint;

        vtkIdType vMeshIdApexStart;
        vtkIdType vMeshIdApexEnd;
        deque< vtkIdType > vMeshIdGroup;
        deque< deque< vtkIdType > > vMeshIdSet;

        deque< deque< vtkIdType > >::iterator vThisGroup;
        deque< deque< vtkIdType > >::iterator vNextGroup;

        deque< vtkIdType >::iterator vThisThisPoint;
        deque< vtkIdType >::iterator vThisNextPoint;
        deque< vtkIdType >::iterator vNextThisPoint;
        deque< vtkIdType >::iterator vNextNextPoint;

        vtkSmartPointer< vtkTriangle > vTriangle;
        vtkSmartPointer< vtkQuad > vQuad;

        //create apex start point id
        LocalToGlobal( anApexStart, tPoint );
        vMeshIdApexStart = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );

        //create apex end point id
        LocalToGlobal( anApexEnd, tPoint );
        vMeshIdApexEnd = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );

        //create mesh point ids
        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
        {
            vMeshIdGroup.clear();
            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
            {
                LocalToGlobal( *tGroupIt, tPoint );
                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
            }
            vMeshIdGroup.push_back( vMeshIdGroup.front() );
            vMeshIdSet.push_back( vMeshIdGroup );
        }

        //create start cap cells
        vThisThisPoint = vMeshIdSet.front().begin();
        vThisNextPoint = ++(vMeshIdSet.front().begin());
        while( vThisNextPoint != vMeshIdSet.front().end() )
        {
            vTriangle = vtkSmartPointer< vtkTriangle >::New();
            vTriangle->GetPointIds()->SetId( 0, *vThisThisPoint );
            vTriangle->GetPointIds()->SetId( 1, *vThisNextPoint );
            vTriangle->GetPointIds()->SetId( 2, vMeshIdApexStart );
            fCells->InsertNextCell( vTriangle );
            fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
            ++vThisThisPoint;
            ++vThisNextPoint;
        }

        //create hull cells
        vThisGroup = vMeshIdSet.begin();
        vNextGroup = ++(vMeshIdSet.begin());
        while( vNextGroup != vMeshIdSet.end() )
        {
            vThisThisPoint = vThisGroup->begin();
            vThisNextPoint = ++(vThisGroup->begin());
            vNextThisPoint = vNextGroup->begin();
            vNextNextPoint = ++(vNextGroup->begin());

            while( vNextNextPoint != vNextGroup->end() )
            {
                vQuad = vtkSmartPointer< vtkQuad >::New();
                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
                fCells->InsertNextCell( vQuad );
                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
                ++vThisThisPoint;
                ++vThisNextPoint;
                ++vNextThisPoint;
                ++vNextNextPoint;
            }

            ++vThisGroup;
            ++vNextGroup;
        }

        //create end cap cells
        vThisThisPoint = vMeshIdSet.back().begin();
        vThisNextPoint = ++(vMeshIdSet.back().begin());
        while( vThisNextPoint != vMeshIdSet.back().end() )
        {
            vTriangle = vtkSmartPointer< vtkTriangle >::New();
            vTriangle->GetPointIds()->SetId( 0, *vThisThisPoint );
            vTriangle->GetPointIds()->SetId( 1, *vThisNextPoint );
            vTriangle->GetPointIds()->SetId( 2, vMeshIdApexEnd );
            fCells->InsertNextCell( vTriangle );
            fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
            ++vThisThisPoint;
            ++vThisNextPoint;
        }

        return;
    }
void KGVTKGeometryPainter::ClosedShellMeshToVTK( const ShellMesh& aMesh )
{
	std::cout << "sending shell mesh to VTK" << std::endl;
    	if ( aMesh.fData.size() == 0 )
    	{
    		vismsg( eWarning ) <<"mesh has size of zero, check your geometry"<<eom;
    	}

        //object allocation
    KThreeVector tPoint;

    deque< vtkIdType > vMeshIdGroup;
    deque< deque< vtkIdType > > vMeshIdSet;

    deque< deque< vtkIdType > >::iterator vThisGroup;
    deque< deque< vtkIdType > >::iterator vNextGroup;

    deque< vtkIdType >::iterator vThisThisPoint;
    deque< vtkIdType >::iterator vThisNextPoint;
    deque< vtkIdType >::iterator vNextThisPoint;
    deque< vtkIdType >::iterator vNextNextPoint;

    vtkSmartPointer< vtkQuad > vQuad;

        //create mesh point ids
    for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
    {
        vMeshIdGroup.clear();
        for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
        {
            LocalToGlobal( *tGroupIt, tPoint );
            vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
        }
        
        vMeshIdSet.push_back( vMeshIdGroup );
    }
    vMeshIdSet.push_back( vMeshIdSet.front() );

        //create hull cells
    vThisGroup = vMeshIdSet.begin();
    vNextGroup = ++(vMeshIdSet.begin());
    while( vNextGroup != vMeshIdSet.end() )
    {
        vThisThisPoint = vThisGroup->begin();
        vThisNextPoint = ++(vThisGroup->begin());
        vNextThisPoint = vNextGroup->begin();
        vNextNextPoint = ++(vNextGroup->begin());

        while( vNextNextPoint != vNextGroup->end() )
        {
            vQuad = vtkSmartPointer< vtkQuad >::New();
            vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
            vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
            vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
            vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
            fCells->InsertNextCell( vQuad );
            fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
            ++vThisThisPoint;
            ++vThisNextPoint;
            ++vNextThisPoint;
            ++vNextNextPoint;
        }

        ++vThisGroup;
        ++vNextGroup;
    }

    return;
} 
void KGVTKGeometryPainter::ShellMeshToVTK( const ShellMesh& aMesh )
{
    std::cout << "sending shell mesh to VTK" << std::endl;
        if ( aMesh.fData.size() == 0 )
        {
            vismsg( eWarning ) <<"mesh has size of zero, check your geometry"<<eom;
        }

        //object allocation
    KThreeVector tPoint;

    deque< vtkIdType > vMeshIdGroup;
    deque< deque< vtkIdType > > vMeshIdSet;

    deque< deque< vtkIdType > >::iterator vThisGroup;
    deque< deque< vtkIdType > >::iterator vNextGroup;

    deque< vtkIdType >::iterator vThisThisPoint;
    deque< vtkIdType >::iterator vThisNextPoint;
    deque< vtkIdType >::iterator vNextThisPoint;
    deque< vtkIdType >::iterator vNextNextPoint;

    vtkSmartPointer< vtkQuad > vQuad;

        //create mesh point ids
    for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
    {
        vMeshIdGroup.clear();
        for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
        {
            LocalToGlobal( *tGroupIt, tPoint );
            vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
        }
        
        vMeshIdSet.push_back( vMeshIdGroup );
    }

        //create hull cells
    vThisGroup = vMeshIdSet.begin();
    vNextGroup = ++(vMeshIdSet.begin());
    while( vNextGroup != vMeshIdSet.end() )
    {
        vThisThisPoint = vThisGroup->begin();
        vThisNextPoint = ++(vThisGroup->begin());
        vNextThisPoint = vNextGroup->begin();
        vNextNextPoint = ++(vNextGroup->begin());

        while( vNextNextPoint != vNextGroup->end() )
        {
            vQuad = vtkSmartPointer< vtkQuad >::New();
            vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
            vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
            vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
            vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
            fCells->InsertNextCell( vQuad );
            fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
            ++vThisThisPoint;
            ++vThisNextPoint;
            ++vNextThisPoint;
            ++vNextNextPoint;
        }

        ++vThisGroup;
        ++vNextGroup;
    }

    return;
}
   void KGVTKGeometryPainter::TorusMeshToVTK( const TorusMesh& aMesh )
    {
        //object allocation
        KThreeVector tPoint;

        deque< vtkIdType > vMeshIdGroup;
        deque< deque< vtkIdType > > vMeshIdSet;

        deque< deque< vtkIdType > >::iterator vThisGroup;
        deque< deque< vtkIdType > >::iterator vNextGroup;

        deque< vtkIdType >::iterator vThisThisPoint;
        deque< vtkIdType >::iterator vThisNextPoint;
        deque< vtkIdType >::iterator vNextThisPoint;
        deque< vtkIdType >::iterator vNextNextPoint;

        vtkSmartPointer< vtkQuad > vQuad;

        //create mesh point ids
        for( TubeMesh::SetCIt tSetIt = aMesh.fData.begin(); tSetIt != aMesh.fData.end(); tSetIt++ )
        {
            vMeshIdGroup.clear();
            for( TubeMesh::GroupCIt tGroupIt = tSetIt->begin(); tGroupIt != tSetIt->end(); tGroupIt++ )
            {
                LocalToGlobal( *tGroupIt, tPoint );
                vMeshIdGroup.push_back( fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() ) );
            }
            vMeshIdGroup.push_back( vMeshIdGroup.front() );
            vMeshIdSet.push_back( vMeshIdGroup );
        }
        vMeshIdSet.push_back( vMeshIdSet.front() );

        //create hull cells
        vThisGroup = vMeshIdSet.begin();
        vNextGroup = ++(vMeshIdSet.begin());
        while( vNextGroup != vMeshIdSet.end() )
        {
            vThisThisPoint = vThisGroup->begin();
            vThisNextPoint = ++(vThisGroup->begin());
            vNextThisPoint = vNextGroup->begin();
            vNextNextPoint = ++(vNextGroup->begin());

            while( vNextNextPoint != vNextGroup->end() )
            {
                vQuad = vtkSmartPointer< vtkQuad >::New();
                vQuad->GetPointIds()->SetId( 0, *vThisThisPoint );
                vQuad->GetPointIds()->SetId( 1, *vNextThisPoint );
                vQuad->GetPointIds()->SetId( 2, *vNextNextPoint );
                vQuad->GetPointIds()->SetId( 3, *vThisNextPoint );
                fCells->InsertNextCell( vQuad );
                fColors->InsertNextTuple4( fCurrentData->GetColor().GetRed(), fCurrentData->GetColor().GetGreen(), fCurrentData->GetColor().GetBlue(), fCurrentData->GetColor().GetOpacity() );
                ++vThisThisPoint;
                ++vThisNextPoint;
                ++vNextThisPoint;
                ++vNextNextPoint;
            }

            ++vThisGroup;
            ++vNextGroup;
        }

        return;
    }

}
