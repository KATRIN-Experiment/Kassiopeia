#include "KGVTKMeshIntersectionTester.hh"
#include "KGVisualizationMessage.hh"
#include "KGMesher.hh"


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

using namespace std;

namespace KGeoBag
{
    KGVTKMeshIntersectionTester::KGVTKMeshIntersectionTester():
            fContainer(),
            fTree(),
            fIntersectionCalculator(),
            fSampleCount( 1000 ),
            fSampleColor( 255, 0, 0 ),
            fPointColor( 0, 255, 0 ),
            fUnintersectedLineColor( 0, 0, 255 ),
            fIntersectedLineColor( 0, 255, 0 ),
            fVertexSize( 0.01 ),
            fLineSize( 0.01 ),
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

    KGVTKMeshIntersectionTester::~KGVTKMeshIntersectionTester()
    {
    }

    void KGVTKMeshIntersectionTester::Construct()
    {
        //we iterate over the surfaces and collect mesh elements
        KGMeshElementCollector tCollector;
        tCollector.SetMeshElementContainer(&fContainer);
        vector< KGSurface* >::iterator tSurfaceIt;
        for( tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++ )
        {
            (*tSurfaceIt)->AcceptNode( &tCollector );
        }

        vector< KGSpace* >::iterator tSpaceIt;
        for( tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); tSpaceIt++ )
        {
            (*tSpaceIt)->AcceptNode( &tCollector );
        }

        KGNavigableMeshTreeBuilder tBuilder;
        tBuilder.SetNavigableMeshElementContainer(&fContainer);
        tBuilder.SetTree(&fTree);
        tBuilder.ConstructTree();

        KGCube<KGMESH_DIM>* world_cube =
        KGObjectRetriever<KGMeshNavigationNodeObjects, KGCube<KGMESH_DIM> >::GetNodeObject(fTree.GetRootNode());

        fBoundingBall.SetCenter( world_cube->GetCenter() );
        double r = std::sqrt( 3.0*(world_cube->GetLength()/2.0)*(world_cube->GetLength()/2.0) );
        fBoundingBall.SetRadius(1.1*r);
    }

    void KGVTKMeshIntersectionTester::Render()
    {
        Construct();

        KRandom& tRandom = KRandom::GetInstance();
        KThreeVector tStartPoint;
        KThreeVector tEndPoint;
        KThreeVector tIntersectionPoint;
        KGPoint<KGMESH_DIM> ori = fBoundingBall.GetCenter();
        KThreeVector tOrigin( ori[0], ori[1], ori[2] );
        fIntersectionCalculator.SetMeshElementContainer(&fContainer);

        vtkIdType vStartPointId;
        vtkIdType vEndPointId;
        vtkIdType vIntersectionPointId;

        vtkSmartPointer< vtkLine > vLine;

        for( unsigned int tIndex = 0; tIndex < fSampleCount; tIndex++ )
        {
            //std::cout<<"sample = "<<tIndex<<std::endl;

            //generate two points in the bounding ball volume
            double r = fBoundingBall.GetRadius();
            double r2 = r*r;
            double x1, y1, z1, x2, y2, z2;
            do
            {
                x1 = tRandom.Uniform( -1.0*r, r );
                y1 = tRandom.Uniform( -1.0*r, r );
                z1 = tRandom.Uniform( -1.0*r, r );
            }
            while( x1*x1 + y1*y1 + z1*z1 > r2  );

            do
            {
                x2 = tRandom.Uniform( -1.0*r, r );
                y2 = tRandom.Uniform( -1.0*r, r );
                z2 = tRandom.Uniform( -1.0*r, r );
            }
            while( x2*x2 + y2*y2 + z2*z2 > r2  );

            tStartPoint = tOrigin + KThreeVector(x1,y1,z1);
            tEndPoint = tOrigin  + KThreeVector(x2,y2,z2);

            fIntersectionCalculator.SetLineSegment(tStartPoint, tEndPoint);
            fIntersectionCalculator.ApplyAction(fTree.GetRootNode());

            if( fIntersectionCalculator.HasIntersectionWithMesh() )
            {
                KThreeVector tIntersectionPoint = fIntersectionCalculator.GetIntersection();

                vIntersectionPointId = fPoints->InsertNextPoint( tIntersectionPoint.X(), tIntersectionPoint.Y(), tIntersectionPoint.Z() );
                fColors->InsertNextTuple3( fIntersectedLineColor.GetRed(), fIntersectedLineColor.GetGreen(), fIntersectedLineColor.GetBlue() );
                fPointCells->InsertNextCell( 1, &vIntersectionPointId );

                vStartPointId = fPoints->InsertNextPoint( tStartPoint.X(), tStartPoint.Y(), tStartPoint.Z() );
                fColors->InsertNextTuple3( fIntersectedLineColor.GetRed(), fIntersectedLineColor.GetGreen(), fIntersectedLineColor.GetBlue() );
                fPointCells->InsertNextCell( 1, &vStartPointId );

                vEndPointId = fPoints->InsertNextPoint( tEndPoint.X(), tEndPoint.Y(), tEndPoint.Z() );
                fColors->InsertNextTuple3( fIntersectedLineColor.GetRed(), fIntersectedLineColor.GetGreen(), fIntersectedLineColor.GetBlue() );
                fPointCells->InsertNextCell( 1, &vEndPointId );

                vLine = vtkSmartPointer< vtkLine >::New();
                vLine->GetPointIds()->SetId( 0, vStartPointId );
                vLine->GetPointIds()->SetId( 1, vEndPointId);
                fLineCells->InsertNextCell( vLine );
            }
        }

        fActor->GetProperty()->SetPointSize( fVertexSize );
        fActor->GetProperty()->SetLineWidth( fLineSize );

        return;
    }

    void KGVTKMeshIntersectionTester::Display()
    {
        if( fDisplayEnabled == true )
        {
            vtkSmartPointer< vtkRenderer > vRenderer = fWindow->GetRenderer();
            vRenderer->AddActor( fActor );
        }
        return;
    }

    void KGVTKMeshIntersectionTester::Write()
    {
        if( fWriteEnabled == true )
        {
            string tFileName = string( OUTPUT_DEFAULT_DIR ) + string( "/" ) + fName + string( ".vtp" );

            vismsg( eNormal ) << "vtk mesh intersection tester <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells() << "> cells to file <" << tFileName << ">" << eom;

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
