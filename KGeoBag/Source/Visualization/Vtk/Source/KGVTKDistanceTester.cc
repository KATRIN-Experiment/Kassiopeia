#include "KGVTKDistanceTester.hh"
#include "KGVisualizationMessage.hh"

#include "KFile.h"
using katrin::KFile;

#include "KConst.h"
using katrin::KConst;

#include "KRandom.h"
using katrin::KRandom;

#include "vtkProperty.h"
#include "vtkCellData.h"

#include <cmath>

using namespace std;

namespace KGeoBag
{

    KGVTKDistanceTester::KGVTKDistanceTester() :
            fSampleDiskOrigin( KThreeVector::sZero ),
            fSampleDiskNormal( KThreeVector::sZUnit ),
            fSampleDiskRadius( 1. ),
            fSampleCount( 0 ),
            fVertexSize( 0 ),
            fPoints( vtkSmartPointer< vtkPoints >::New() ),
            fValues( vtkSmartPointer< vtkDoubleArray >::New() ),
            fCells( vtkSmartPointer< vtkCellArray >::New() ),
            fPolyData( vtkSmartPointer< vtkPolyData >::New() ),
            fTable( vtkSmartPointer< vtkLookupTable >::New() ),
            fMapper( vtkSmartPointer< vtkPolyDataMapper >::New() ),
            fActor( vtkSmartPointer< vtkActor >::New() )
    {
        fValues->SetNumberOfComponents( 1 );
        fPolyData->SetPoints( fPoints );
        fPolyData->SetVerts( fCells );
        fPolyData->GetCellData()->SetScalars( fValues );
        fTable->SetNumberOfTableValues( 256 );
        fTable->SetHueRange( 0.000, 0.667 );
        fTable->SetRampToLinear();
        fTable->SetVectorModeToMagnitude();
        fTable->Build();
#ifdef VTK6
        fMapper->SetInputData( fPolyData );
#else
        fMapper->SetInput( fPolyData );
#endif
        fMapper->SetScalarModeToUseCellData();
        fMapper->SetLookupTable( fTable );
        fMapper->SetColorModeToMapScalars();
        fMapper->ScalarVisibilityOn();
        fActor->SetMapper( fMapper );
    }

    KGVTKDistanceTester::~KGVTKDistanceTester()
    {
    }

    void KGVTKDistanceTester::Render()
    {
        KRandom& tRandom = KRandom::GetInstance();
        KThreeVector tUnitOne = fSampleDiskNormal.Orthogonal().Unit();
        KThreeVector tUnitTwo = fSampleDiskNormal.Orthogonal().Cross( fSampleDiskNormal ).Unit();
        double tRadius;
        double tPhi;

        KThreeVector tPoint;
        KThreeVector tNearestPoint;

        vtkIdType vPointId;

        for( unsigned int tIndex = 0; tIndex < fSampleCount; tIndex++ )
        {
            tRadius = fSampleDiskRadius * sqrt( tRandom.Uniform( 0., 1. ) );
            tPhi = tRandom.Uniform( 0., 2. * KConst::Pi() );
            tPoint = fSampleDiskOrigin + tRadius * (cos( tPhi ) * tUnitOne + sin( tPhi ) * tUnitTwo);

            for( vector< const KGSurface* >::iterator tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++ )
            {
                tNearestPoint = (*tSurfaceIt)->Point( tPoint );

                vPointId = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );
                fValues->InsertNextTuple1( (tPoint - tNearestPoint).Magnitude() );
                fCells->InsertNextCell( 1, &vPointId );
            }

            for( vector< const KGSpace* >::iterator tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); tSpaceIt++ )
            {
                tNearestPoint = (*tSpaceIt)->Point( tPoint );

                vPointId = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );
                fValues->InsertNextTuple1( (tPoint - tNearestPoint).Magnitude() );
                fCells->InsertNextCell( 1, &vPointId );
            }
        }

        fActor->GetProperty()->SetPointSize( fVertexSize );

        return;
    }

    void KGVTKDistanceTester::Display()
    {
        if( fDisplayEnabled == true )
        {
            vtkSmartPointer< vtkRenderer > vRenderer = fWindow->GetRenderer();
            vRenderer->AddActor( fActor );
        }
        return;
    }

    void KGVTKDistanceTester::Write()
    {
        if( fWriteEnabled == true )
        {
            string tFileName = string( OUTPUT_DEFAULT_DIR ) + string( "/" ) + GetName() + string( ".vtp" );

            vismsg( eNormal ) << "vtk distance tester <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells() << "> cells to file <" << tFileName << ">" << eom;

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
