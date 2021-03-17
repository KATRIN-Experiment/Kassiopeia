#include "KGVTKNormalTester.hh"

#include "KConst.h"
#include "KFile.h"
#include "KGVisualizationMessage.hh"
#include "KRandom.h"
using katrin::KRandom;

#include "vtkLine.h"
#include "vtkPointData.h"
#include "vtkProperty.h"

#include <cmath>

using namespace std;

namespace KGeoBag
{

KGVTKNormalTester::KGVTKNormalTester() :
    fSampleDiskOrigin(KThreeVector::sZero),
    fSampleDiskNormal(KThreeVector::sZUnit),
    fSampleDiskRadius(1.),
    fSampleCount(0),
    fSampleColor(255, 0, 0),
    fPointColor(0, 255, 0),
    fNormalColor(0, 0, 255),
    fNormalLength(0.01),
    fVertexSize(0.001),
    fLineSize(0.001),
    fPoints(vtkSmartPointer<vtkPoints>::New()),
    fColors(vtkSmartPointer<vtkUnsignedCharArray>::New()),
    fPointCells(vtkSmartPointer<vtkCellArray>::New()),
    fLineCells(vtkSmartPointer<vtkCellArray>::New()),
    fPolyData(vtkSmartPointer<vtkPolyData>::New()),
    fMapper(vtkSmartPointer<vtkPolyDataMapper>::New()),
    fActor(vtkSmartPointer<vtkActor>::New())
{
    fColors->SetNumberOfComponents(3);
    fPolyData->SetPoints(fPoints);
    fPolyData->SetVerts(fPointCells);
    fPolyData->SetLines(fLineCells);
    fPolyData->GetPointData()->SetScalars(fColors);
#ifdef VTK6
    fMapper->SetInputData(fPolyData);
#else
    fMapper->SetInput(fPolyData);
#endif
    fMapper->SetScalarModeToUsePointData();
    fActor->SetMapper(fMapper);
}

KGVTKNormalTester::~KGVTKNormalTester() = default;

void KGVTKNormalTester::Render()
{
    KRandom& tRandom = KRandom::GetInstance();
    KThreeVector tUnitOne = fSampleDiskNormal.Orthogonal().Unit();
    KThreeVector tUnitTwo = fSampleDiskNormal.Orthogonal().Cross(fSampleDiskNormal).Unit();
    double tRadius;
    double tPhi;

    KThreeVector tPoint;
    KThreeVector tNearest;
    KThreeVector tNormal;

    vtkIdType vPointId;
    vtkIdType vNearestId;
    vtkIdType vNormalId;
    vtkSmartPointer<vtkLine> vLine;

    for (unsigned int tIndex = 0; tIndex < fSampleCount; tIndex++) {
        tRadius = fSampleDiskRadius * sqrt(tRandom.Uniform(0., 1.));
        tPhi = tRandom.Uniform(0., 2. * katrin::KConst::Pi());
        tPoint = fSampleDiskOrigin + tRadius * (cos(tPhi) * tUnitOne + sin(tPhi) * tUnitTwo);

        for (auto& surface : fSurfaces) {
            tNearest = surface->Point(tPoint);
            tNormal = surface->Normal(tPoint);
            tNormal = tNearest + fNormalLength * tNormal;

            vPointId = fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z());
            fColors->InsertNextTuple3(fSampleColor.GetRed(), fSampleColor.GetGreen(), fSampleColor.GetBlue());
            fPointCells->InsertNextCell(1, &vPointId);

            vNearestId = fPoints->InsertNextPoint(tNearest.X(), tNearest.Y(), tNearest.Z());
            fColors->InsertNextTuple3(fPointColor.GetRed(), fPointColor.GetGreen(), fPointColor.GetBlue());
            fPointCells->InsertNextCell(1, &vNearestId);

            vNormalId = fPoints->InsertNextPoint(tNormal.X(), tNormal.Y(), tNormal.Z());
            fColors->InsertNextTuple3(fNormalColor.GetRed(), fNormalColor.GetGreen(), fNormalColor.GetBlue());
            fPointCells->InsertNextCell(1, &vNormalId);

            vLine = vtkSmartPointer<vtkLine>::New();
            vLine->GetPointIds()->SetId(0, vPointId);
            vLine->GetPointIds()->SetId(1, vNearestId);
            fLineCells->InsertNextCell(vLine);

            vLine = vtkSmartPointer<vtkLine>::New();
            vLine->GetPointIds()->SetId(0, vNearestId);
            vLine->GetPointIds()->SetId(1, vNormalId);
            fLineCells->InsertNextCell(vLine);
        }

        for (auto& space : fSpaces) {
            tNearest = space->Point(tPoint);
            tNormal = space->Normal(tPoint);
            tNormal = tNearest + fNormalLength * tNormal;

            vPointId = fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z());
            fColors->InsertNextTuple3(fSampleColor.GetRed(), fSampleColor.GetGreen(), fSampleColor.GetBlue());
            fPointCells->InsertNextCell(1, &vPointId);

            vNearestId = fPoints->InsertNextPoint(tNearest.X(), tNearest.Y(), tNearest.Z());
            fColors->InsertNextTuple3(fPointColor.GetRed(), fPointColor.GetGreen(), fPointColor.GetBlue());
            fPointCells->InsertNextCell(1, &vNearestId);

            vNormalId = fPoints->InsertNextPoint(tNormal.X(), tNormal.Y(), tNormal.Z());
            fColors->InsertNextTuple3(fNormalColor.GetRed(), fNormalColor.GetGreen(), fNormalColor.GetBlue());
            fPointCells->InsertNextCell(1, &vNormalId);

            vLine = vtkSmartPointer<vtkLine>::New();
            vLine->GetPointIds()->SetId(0, vPointId);
            vLine->GetPointIds()->SetId(1, vNearestId);
            fLineCells->InsertNextCell(vLine);

            vLine = vtkSmartPointer<vtkLine>::New();
            vLine->GetPointIds()->SetId(0, vNearestId);
            vLine->GetPointIds()->SetId(1, vNormalId);
            fLineCells->InsertNextCell(vLine);
        }
    }

    fActor->GetProperty()->SetPointSize(fVertexSize);
    fActor->GetProperty()->SetLineWidth(fLineSize);

    return;
}

void KGVTKNormalTester::Display()
{
    if (fDisplayEnabled == true) {
        vtkSmartPointer<vtkRenderer> vRenderer = fWindow->GetRenderer();
        vRenderer->AddActor(fActor);
    }
    return;
}

void KGVTKNormalTester::Write()
{
    if (fWriteEnabled == true) {
        string tFileName = string(OUTPUT_DEFAULT_DIR) + string("/") + GetName() + string(".vtp");

        vismsg(eNormal) << "vtk normal tester <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells()
                        << "> cells to file <" << tFileName << ">" << eom;

        vtkSmartPointer<vtkXMLPolyDataWriter> vWriter = fWindow->GetWriter();
        vWriter->SetFileName(tFileName.c_str());
        vWriter->SetDataModeToBinary();
#ifdef VTK6
        vWriter->SetInputData(fPolyData);
#else
        vWriter->SetInput(fPolyData);
#endif
        vWriter->Write();
    }
    return;
}

}  // namespace KGeoBag
