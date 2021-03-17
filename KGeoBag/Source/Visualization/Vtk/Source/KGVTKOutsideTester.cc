#include "KGVTKOutsideTester.hh"

#include "KConst.h"
#include "KFile.h"
#include "KGVisualizationMessage.hh"
#include "KRandom.h"
using katrin::KRandom;

#include "vtkPointData.h"
#include "vtkProperty.h"

#include <cmath>

using namespace std;

namespace KGeoBag
{

KGVTKOutsideTester::KGVTKOutsideTester() :
    fSampleDiskOrigin(KThreeVector::sZero),
    fSampleDiskNormal(KThreeVector::sZUnit),
    fSampleDiskRadius(1.),
    fSampleCount(0),
    fInsideColor(0, 255, 0),
    fOutsideColor(255, 0, 0),
    fVertexSize(0.001),
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

KGVTKOutsideTester::~KGVTKOutsideTester() = default;

void KGVTKOutsideTester::Render()
{
    KRandom& tRandom = KRandom::GetInstance();
    KThreeVector tUnitOne = fSampleDiskNormal.Orthogonal().Unit();
    KThreeVector tUnitTwo = fSampleDiskNormal.Orthogonal().Cross(fSampleDiskNormal).Unit();
    double tRadius;
    double tPhi;

    KThreeVector tPoint;
    bool tOutside;

    vtkIdType vPointId;

    for (unsigned int tIndex = 0; tIndex < fSampleCount; tIndex++) {
        tRadius = fSampleDiskRadius * sqrt(tRandom.Uniform(0., 1.));
        tPhi = tRandom.Uniform(0., 2. * katrin::KConst::Pi());
        tPoint = fSampleDiskOrigin + tRadius * (cos(tPhi) * tUnitOne + sin(tPhi) * tUnitTwo);

        for (auto& surface : fSurfaces) {
            tOutside = surface->Above(tPoint);

            vPointId = fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z());

            if (tOutside == true) {
                fColors->InsertNextTuple3(fOutsideColor.GetRed(), fOutsideColor.GetGreen(), fOutsideColor.GetBlue());
            }
            else {
                fColors->InsertNextTuple3(fInsideColor.GetRed(), fInsideColor.GetGreen(), fInsideColor.GetBlue());
            }

            fPointCells->InsertNextCell(1, &vPointId);
        }

        for (auto& space : fSpaces) {
            tOutside = space->Outside(tPoint);

            vPointId = fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z());

            if (tOutside == true) {
                fColors->InsertNextTuple3(fOutsideColor.GetRed(), fOutsideColor.GetGreen(), fOutsideColor.GetBlue());
            }
            else {
                fColors->InsertNextTuple3(fInsideColor.GetRed(), fInsideColor.GetGreen(), fInsideColor.GetBlue());
            }

            fPointCells->InsertNextCell(1, &vPointId);
        }
    }

    fActor->GetProperty()->SetPointSize(fVertexSize);

    return;
}

void KGVTKOutsideTester::Display()
{
    if (fDisplayEnabled == true) {
        vtkSmartPointer<vtkRenderer> vRenderer = fWindow->GetRenderer();
        vRenderer->AddActor(fActor);
    }
    return;
}

void KGVTKOutsideTester::Write()
{
    if (fWriteEnabled == true) {
        string tFileName = string(OUTPUT_DEFAULT_DIR) + string("/") + GetName() + string(".vtp");

        vismsg(eNormal) << "vtk outside tester <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells()
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
