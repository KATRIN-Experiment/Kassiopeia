#include "KGVTKPointTester.hh"
#include "KGVisualizationMessage.hh"

#include "KFile.h"
using katrin::KFile;

#include "KConst.h"
using katrin::KConst;

#include "KRandom.h"
using katrin::KRandom;

#include "vtkProperty.h"
#include "vtkPointData.h"
#include "vtkLine.h"

#include <cmath>

namespace KGeoBag
{

    KGVTKPointTester::KGVTKPointTester() :
            fSampleDiskOrigin( KThreeVector::sZero ),
            fSampleDiskNormal( KThreeVector::sZUnit ),
            fSampleDiskRadius( 1. ),
            fSampleCount( 0 ),
            fSampleColor( 255, 0, 0 ),
            fPointColor( 0, 255, 0 ),
            fVertexSize( 0.001 ),
            fLineSize( 0.001 ),
            fPoints( vtkSmartPointer< vtkPoints >::New() ),
            fColors( vtkSmartPointer< vtkUnsignedCharArray >::New() ),
            fPointCells( vtkSmartPointer< vtkCellArray >::New() ),
            fLineCells( vtkSmartPointer< vtkCellArray >::New() ),
            fPolyData( vtkSmartPointer< vtkPolyData >::New() ),
            fMapper( vtkSmartPointer< vtkPolyDataMapper >::New() ),
            fActor( vtkSmartPointer< vtkActor >::New() )
    {
        fColors->SetNumberOfComponents( 3 );
        fPolyData->SetPoints( fPoints );
        fPolyData->SetVerts( fPointCells );
        fPolyData->SetLines( fLineCells );
        fPolyData->GetPointData()->SetScalars( fColors );
#ifdef VTK6
        fMapper->SetInputData( fPolyData );
#else
        fMapper->SetInput( fPolyData );
#endif
        fMapper->SetScalarModeToUsePointData();
        fActor->SetMapper( fMapper );
    }

    KGVTKPointTester::~KGVTKPointTester()
    {
    }

    void KGVTKPointTester::Render()
    {
        KRandom& tRandom = KRandom::GetInstance();
        KThreeVector tUnitOne = fSampleDiskNormal.Orthogonal().Unit();
        KThreeVector tUnitTwo = fSampleDiskNormal.Orthogonal().Cross( fSampleDiskNormal ).Unit();
        double tRadius;
        double tPhi;

        KThreeVector tPoint;
        KThreeVector tNearestPoint;

        vtkIdType vPointId;
        vtkIdType vNearestId;
        vtkSmartPointer< vtkLine > vLine;

        for( unsigned int tIndex = 0; tIndex < fSampleCount; tIndex++ )
        {
            tRadius = fSampleDiskRadius * sqrt( tRandom.Uniform( 0., 1. ) );
            tPhi = tRandom.Uniform( 0., 2. * KConst::Pi() );
            tPoint = fSampleDiskOrigin + tRadius * (cos( tPhi ) * tUnitOne + sin( tPhi ) * tUnitTwo);

            for( vector< const KGSurface* >::iterator tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++ )
            {
                tNearestPoint = (*tSurfaceIt)->Point( tPoint );

                vPointId = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );
                fColors->InsertNextTuple3( fSampleColor.GetRed(), fSampleColor.GetGreen(), fSampleColor.GetBlue() );
                fPointCells->InsertNextCell( 1, &vPointId );

                vNearestId = fPoints->InsertNextPoint( tNearestPoint.X(), tNearestPoint.Y(), tNearestPoint.Z() );
                fColors->InsertNextTuple3( fPointColor.GetRed(), fPointColor.GetGreen(), fPointColor.GetBlue() );
                fPointCells->InsertNextCell( 1, &vNearestId );

                vLine = vtkSmartPointer< vtkLine >::New();
                vLine->GetPointIds()->SetId( 0, vPointId );
                vLine->GetPointIds()->SetId( 1, vNearestId );
                fLineCells->InsertNextCell( vLine );
            }

            for( vector< const KGSpace* >::iterator tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); tSpaceIt++ )
            {
                tNearestPoint = (*tSpaceIt)->Point( tPoint );

                vPointId = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );
                fColors->InsertNextTuple3( fSampleColor.GetRed(), fSampleColor.GetGreen(), fSampleColor.GetBlue() );
                fPointCells->InsertNextCell( 1, &vPointId );

                vNearestId = fPoints->InsertNextPoint( tNearestPoint.X(), tNearestPoint.Y(), tNearestPoint.Z() );
                fColors->InsertNextTuple3( fPointColor.GetRed(), fPointColor.GetGreen(), fPointColor.GetBlue() );
                fPointCells->InsertNextCell( 1, &vNearestId );

                vLine = vtkSmartPointer< vtkLine >::New();
                vLine->GetPointIds()->SetId( 0, vPointId );
                vLine->GetPointIds()->SetId( 1, vNearestId );
                fLineCells->InsertNextCell( vLine );
            }
        }

        fActor->GetProperty()->SetPointSize( fVertexSize );
        fActor->GetProperty()->SetLineWidth( fLineSize );

        return;
    }

    void KGVTKPointTester::Display()
    {
        if( fDisplayEnabled == true )
        {
            vtkSmartPointer< vtkRenderer > vRenderer = fWindow->GetRenderer();
            vRenderer->AddActor( fActor );
        }
        return;
    }

    void KGVTKPointTester::Write()
    {
        if( fWriteEnabled == true )
        {
            string tFileName = string( OUTPUT_DEFAULT_DIR ) + string( "/" ) + fName + string( ".vtp" );

            vismsg( eNormal ) << "vtk point tester <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells() << "> cells to file <" << tFileName << ">" << eom;

            vtkSmartPointer< vtkXMLPolyDataWriter > vWriter = fWindow->GetWriter();
            vWriter->SetFileName( tFileName.c_str() );
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

}
