#include "KGVTKRandomPointTester.hh"
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

	KGVTKRandomPointTester::KGVTKRandomPointTester() :
            fSampleColor( 0, 255, 0 ),
            fVertexSize( 0.001 ),
//            fSampleCount( 0 ),
            fPoints( vtkSmartPointer< vtkPoints >::New() ),
            fColors( vtkSmartPointer< vtkUnsignedCharArray >::New() ),
            fCells( vtkSmartPointer< vtkCellArray >::New() ),
            fPolyData( vtkSmartPointer< vtkPolyData >::New() ),
            fMapper( vtkSmartPointer< vtkPolyDataMapper >::New() ),
            fActor( vtkSmartPointer< vtkActor >::New() )
    {
        fColors->SetNumberOfComponents( 3 );
        fPolyData->SetPoints( fPoints );
        fPolyData->SetVerts( fCells );
        fPolyData->GetPointData()->SetScalars( fColors );
#ifdef VTK6
        fMapper->SetInputData( fPolyData );
#else
        fMapper->SetInput( fPolyData );
#endif
        fMapper->SetScalarModeToUsePointData();
        fActor->SetMapper( fMapper );
    }

	KGVTKRandomPointTester::~KGVTKRandomPointTester()
    {
    }

    void KGVTKRandomPointTester::Render()
    {
        KThreeVector tPoint;

        vtkIdType vPointId;

        for(std::vector<KThreeVector*>::const_iterator p = fSamplePoints.begin();
        		p != fSamplePoints.end(); ++p) {
            tPoint = *(*p);

            vPointId = fPoints->InsertNextPoint( tPoint.X(), tPoint.Y(), tPoint.Z() );
			fColors->InsertNextTuple3( fSampleColor.GetRed(), fSampleColor.GetGreen(), fSampleColor.GetBlue() );
			fCells->InsertNextCell( 1, &vPointId );
        }

        fActor->GetProperty()->SetPointSize( fVertexSize );

        return;
    }

    void KGVTKRandomPointTester::Display()
    {
        if( fDisplayEnabled == true )
        {
            vtkSmartPointer< vtkRenderer > vRenderer = fWindow->GetRenderer();
            vRenderer->AddActor( fActor );
        }
        return;
    }

    void KGVTKRandomPointTester::Write()
    {
        if( fWriteEnabled == true )
        {
            string tFileName = string( OUTPUT_DEFAULT_DIR ) + string( "/" ) + fName + string( ".vtp" );

            vismsg( eNormal ) << "vtk normal tester <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells() << "> cells to file <" << tFileName << ">" << eom;

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
