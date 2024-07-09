#include "KEMVTKViewer.hh"
#include "KEMCoreMessage.hh"

#include <vtkCamera.h>
#include <vtkActor.h>
#include <vtkAppendPolyData.h>
#include <vtkCellArray.h>
#include <vtkCellData.h>
#include <vtkCleanPolyData.h>
#include <vtkDataSetMapper.h>
#include <vtkDiskSource.h>
#include <vtkExtractEdges.h>
#include <vtkImageData.h>
#include <vtkLine.h>
#include <vtkLinearExtrusionFilter.h>
#include <vtkMeshQuality.h>
#include <vtkPointData.h>
#include <vtkPolyDataWriter.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkSTLWriter.h>
#include <vtkTransform.h>
#include <vtkTransformPolyDataFilter.h>
#include <vtkTriangleFilter.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkXMLPPolyDataWriter.h>
#include <vtkXMLPolyDataWriter.h>

namespace KEMField
{
KEMVTKViewer::KEMVTKViewer(KSurfaceContainer& aSurfaceContainer)
{
    fPolyData = vtkSmartPointer<vtkPolyData>::New();

    fPoints = vtkSmartPointer<vtkPoints>::New();

    fCells = vtkSmartPointer<vtkCellArray>::New();

    fArea = vtkSmartPointer<vtkDoubleArray>::New();
    fArea->SetName("Area");

    fLogArea = vtkSmartPointer<vtkDoubleArray>::New();
    fLogArea->SetName("Log Area");

    fModulo = vtkSmartPointer<vtkShortArray>::New();
    fModulo->SetName("Modulo");

    fAspectRatio = vtkSmartPointer<vtkDoubleArray>::New();
    fAspectRatio->SetName("Aspect Ratio");

    //fQuality = vtkSmartPointer<vtkDoubleArray>::New();
    //fQuality->SetName("Cell Quality");

    fCharge = vtkSmartPointer<vtkDoubleArray>::New();
    fCharge->SetName("Charge");

    fChargeDensity = vtkSmartPointer<vtkDoubleArray>::New();
    fChargeDensity->SetName("Charge Density");

    fLogChargeDensity = vtkSmartPointer<vtkDoubleArray>::New();
    fLogChargeDensity->SetName("CD/|CD|*log(1 + |CD*1.e16|)");

    fPotential = vtkSmartPointer<vtkDoubleArray>::New();
    fPotential->SetName("Electric Potential");

    fPermittivity = vtkSmartPointer<vtkDoubleArray>::New();
    fPermittivity->SetName("Electric Permittivity");

    fTriangle = vtkSmartPointer<vtkTriangle>::New();
    fQuad = vtkSmartPointer<vtkQuad>::New();

    fPointCounter = 0;

    fLineSegmentRadiusMin = 1.e-4;
    fLineSegmentPolyApprox = 6;
    fArcPolyApprox = 120;

    for (KSurfaceContainer::iterator it = aSurfaceContainer.begin(); it != aSurfaceContainer.end(); it++) {
        if (! *it)
            continue;
        SetSurfacePrimitive(*it);
        ActOnSurfaceType((*it)->GetID(), *this);
    }

    fPolyData->SetPoints(fPoints);
    fPolyData->SetPolys(fCells);

    // Calculate functions of quality of the elements of a mesh.
    auto qualityFilter = vtkSmartPointer<vtkMeshQuality>::New();
    qualityFilter->SetInputData(fPolyData);
    qualityFilter->SetTriangleQualityMeasureToScaledJacobian();
    qualityFilter->SetQuadQualityMeasureToScaledJacobian();
    qualityFilter->Update();

    vtkDataSet* qualityMesh = qualityFilter->GetOutput();
    fQuality = dynamic_cast<vtkDoubleArray*>(qualityMesh->GetCellData()->GetArray("Quality"));
    fQuality->SetName("Cell Quality");

    fPolyData->GetCellData()->AddArray(fQuality);
    fPolyData->GetCellData()->AddArray(fArea);
    fPolyData->GetCellData()->AddArray(fLogArea);
    fPolyData->GetCellData()->AddArray(fAspectRatio);
    fPolyData->GetCellData()->AddArray(fCharge);
    fPolyData->GetCellData()->AddArray(fChargeDensity);
    fPolyData->GetCellData()->AddArray(fLogChargeDensity);
    fPolyData->GetCellData()->AddArray(fPotential);
    fPolyData->GetCellData()->AddArray(fPermittivity);
    fPolyData->GetCellData()->AddArray(fModulo);
}

void KEMVTKViewer::GenerateGeometryFile(const std::string& fileName)
{
    std::string fileType = fileName.substr(fileName.length()-3);
    if (fileType == "stl") {
        // use triangle filter because STL can only export 3 vertices per object (see vtkSTLWriter documentation)
        auto filter = vtkSmartPointer<vtkTriangleFilter>::New();
#ifdef VTK6
        filter->SetInputData(fPolyData);
#else
        vTriangleFilter->SetInput(fPolyData);
#endif

        auto writer = vtkSmartPointer<vtkSTLWriter>::New();
        //writer->SetFileTypeToASCII();  // binary STL might break import in other programs
        writer->SetFileTypeToBinary();
        writer->SetInputConnection(filter->GetOutputPort());
        writer->SetFileName(fileName.c_str());
        writer->Write();
    }
    else {
        auto writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();

        // writer->SetDataModeToAscii();
        writer->SetDataModeToBinary();
        writer->SetCompressorTypeToZLib();
        writer->SetIdTypeToInt64();
#ifdef VTK6
        writer->SetHeaderTypeToUInt64();
        writer->SetInputData(fPolyData);
#else
        writer->SetInput(fPolyData);
#endif
        writer->SetFileName(fileName.c_str());
        writer->Write();
    }
}

void KEMVTKViewer::ViewGeometry()
{
    fPolyData->GetCellData()->SetScalars(fQuality);

    // Create an actor and mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#ifdef VTK6
    mapper->SetInputData(fPolyData);
#else
    mapper->SetInput(edgeFilter->GetOutput());
#endif
    mapper->SetScalarModeToUseCellData();
    mapper->SetResolveCoincidentTopologyToShiftZBuffer();

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
//    actor->GetProperty()->SetRepresentationToSurface();
//    actor->GetProperty()->SetOpacity(0.5);
    actor->GetProperty()->SetRepresentationToWireframe();

//    // Create edge filter with actor and mapper
//    auto edgeFilter = vtkSmartPointer<vtkExtractEdges>::New();
//#ifdef VTK6
//    edgeFilter->SetInputData(fPolyData);
//#else
//    edgeFilter->SetInput(fPolyData);
//#endif

//    vtkSmartPointer<vtkDataSetMapper> edge_mapper = vtkSmartPointer<vtkDataSetMapper>::New();
//#ifdef VTK6
//    edge_mapper->SetInputConnection(edgeFilter->GetOutputPort());
//#else
//#endif
//    edge_mapper->SetResolveCoincidentTopologyToPolygonOffset();

//    vtkSmartPointer<vtkActor> edge_actor = vtkSmartPointer<vtkActor>::New();
//    edge_actor->SetMapper(edge_mapper);
//    edge_actor->GetProperty()->SetColor(1, 1, 1);

    // Create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(750, 750);
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderer->AddActor(actor);
//    renderer->AddActor(edge_actor);

    renderer->GetActiveCamera()->SetParallelProjection(1);

    renderWindow->Render();

    kem_cout() << "KEMVTKViewer finished; waiting for key press ..." << eom;
    renderWindowInteractor->Start();
    renderWindow->Finalize();
}
}  // namespace KEMField
