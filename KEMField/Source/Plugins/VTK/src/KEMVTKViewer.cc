#include "KEMVTKViewer.hh"
#include "KEMCoreMessage.hh"

#include <vtkCamera.h>

namespace KEMField
{
KEMVTKViewer::KEMVTKViewer(KSurfaceContainer& aSurfaceContainer)
{
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

    fLineSegmentPolyApprox = 3;
    fArcPolyApprox = 128;

    for (KSurfaceContainer::iterator it = aSurfaceContainer.begin(); it != aSurfaceContainer.end(); it++) {
        if (! *it)
            continue;
        SetSurfacePrimitive(*it);
        ActOnSurfaceType((*it)->GetID(), *this);
    }
}

void KEMVTKViewer::GenerateGeometryFile(const std::string& fileName)
{
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(fPoints);
    polydata->SetPolys(fCells);
    polydata->GetCellData()->AddArray(fArea);
    polydata->GetCellData()->AddArray(fLogArea);
    polydata->GetCellData()->AddArray(fModulo);
    polydata->GetCellData()->AddArray(fAspectRatio);
    polydata->GetCellData()->AddArray(fChargeDensity);
    polydata->GetCellData()->AddArray(fLogChargeDensity);
    polydata->GetCellData()->AddArray(fPotential);
    polydata->GetCellData()->AddArray(fPermittivity);

    vtkSmartPointer<vtkXMLPolyDataWriter> writer = vtkSmartPointer<vtkXMLPolyDataWriter>::New();

    // writer->SetDataModeToAscii();
    writer->SetDataModeToBinary();
    writer->SetCompressorTypeToZLib();
    writer->SetIdTypeToInt64();

#ifdef VTK6
    writer->SetHeaderTypeToUInt64();
    writer->SetInputData(polydata);
#else
    writer->SetInput(polydata);
#endif
    writer->SetFileName(fileName.c_str());
    writer->Write();
}

void KEMVTKViewer::ViewGeometry()
{
    vtkSmartPointer<vtkPolyData> polydata = vtkSmartPointer<vtkPolyData>::New();
    polydata->SetPoints(fPoints);
    polydata->SetPolys(fCells);
    polydata->GetCellData()->AddArray(fArea);
    polydata->GetCellData()->AddArray(fLogArea);
    polydata->GetCellData()->AddArray(fModulo);
    polydata->GetCellData()->AddArray(fAspectRatio);
    polydata->GetCellData()->AddArray(fChargeDensity);
    polydata->GetCellData()->AddArray(fLogChargeDensity);
    polydata->GetCellData()->AddArray(fPotential);
    polydata->GetCellData()->AddArray(fPermittivity);

    vtkSmartPointer<vtkTriangleFilter> trifilter = vtkSmartPointer<vtkTriangleFilter>::New();
#ifdef VTK6
    trifilter->SetInputData(polydata);
#else
    trifilter->SetInput(polydata);
#endif

    vtkSmartPointer<vtkStripper> stripper = vtkSmartPointer<vtkStripper>::New();
    stripper->SetInputConnection(trifilter->GetOutputPort());

    // Create an actor and mapper
    vtkSmartPointer<vtkDataSetMapper> mapper = vtkSmartPointer<vtkDataSetMapper>::New();
#ifdef VTK6
    mapper->SetInputConnection(stripper->GetOutputPort());
#else
    mapper->SetInput(stripper->GetOutput());
#endif

    vtkSmartPointer<vtkActor> actor = vtkSmartPointer<vtkActor>::New();
    actor->SetMapper(mapper);
    actor->GetProperty()->SetRepresentationToWireframe();

    // Create a renderer, render window, and interactor
    vtkSmartPointer<vtkRenderer> renderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renderWindow = vtkSmartPointer<vtkRenderWindow>::New();
    renderWindow->SetSize(750, 750);
    renderWindow->AddRenderer(renderer);
    vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    renderWindowInteractor->SetRenderWindow(renderWindow);

    renderer->AddActor(actor);
    renderer->GetActiveCamera()->SetParallelProjection(1);

    renderWindow->Render();

    kem_cout() << "KEMVTKViewer finished; waiting for key press ..." << eom;
    renderWindowInteractor->Start();
    renderWindow->Finalize();
}
}  // namespace KEMField
