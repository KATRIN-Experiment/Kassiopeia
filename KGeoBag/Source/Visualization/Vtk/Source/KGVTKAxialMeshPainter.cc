#include "KGVTKAxialMeshPainter.hh"

#include "KConst.h"
#include "KFile.h"
#include "KGAxialMeshLoop.hh"
#include "KGAxialMeshRing.hh"
#include "KGVisualizationMessage.hh"
#include "vtkCellData.h"
#include "vtkLine.h"
#include "vtkQuad.h"

#include <cmath>

using namespace std;

namespace KGeoBag
{
const unsigned int KGVTKAxialMeshPainter::sArea = 0;
const unsigned int KGVTKAxialMeshPainter::sAspect = 1;
const unsigned int KGVTKAxialMeshPainter::sModulo = 2;

KGVTKAxialMeshPainter::KGVTKAxialMeshPainter() :
    fCurrentElements(nullptr),

    fColorTable(vtkSmartPointer<vtkLookupTable>::New()),
    fAreaData(vtkSmartPointer<vtkDoubleArray>::New()),
    fAspectData(vtkSmartPointer<vtkDoubleArray>::New()),
    fModuloData(vtkSmartPointer<vtkDoubleArray>::New()),
    fPoints(vtkSmartPointer<vtkPoints>::New()),
    fLineCells(vtkSmartPointer<vtkCellArray>::New()),
    fPolyCells(vtkSmartPointer<vtkCellArray>::New()),
    fPolyData(vtkSmartPointer<vtkPolyData>::New()),
    fMapper(vtkSmartPointer<vtkPolyDataMapper>::New()),
    fActor(vtkSmartPointer<vtkActor>::New()),
    fArcCount(128),
    fColorMode(sModulo)
{
    fColorTable->SetNumberOfTableValues(256);
    fColorTable->SetHueRange(0.333, 0.000);
    fColorTable->SetNanColor(127, 127, 127, 255);
    fColorTable->SetRampToLinear();
    fColorTable->SetVectorModeToMagnitude();
    fColorTable->Build();
    fAreaData->SetNumberOfComponents(1);
    fAreaData->SetName("area");
    fAspectData->SetNumberOfComponents(1);
    fAspectData->SetName("aspect");
    fModuloData->SetNumberOfComponents(1);
    fModuloData->SetName("modulo");
    fPolyData->SetPoints(fPoints);
    fPolyData->SetLines(fLineCells);
    fPolyData->SetPolys(fPolyCells);
    fPolyData->GetCellData()->AddArray(fAreaData);
    fPolyData->GetCellData()->AddArray(fAspectData);
    fPolyData->GetCellData()->AddArray(fModuloData);
#ifdef VTK6
    fMapper->SetInputData(fPolyData);
#else
    fMapper->SetInput(fPolyData);
#endif
    fMapper->SetLookupTable(fColorTable);
    fMapper->SetScalarModeToUseCellFieldData();
    fMapper->ScalarVisibilityOn();
    fActor->SetMapper(fMapper);
}
KGVTKAxialMeshPainter::~KGVTKAxialMeshPainter() = default;

void KGVTKAxialMeshPainter::Render()
{
    return;
}
void KGVTKAxialMeshPainter::Display()
{
    if (fDisplayEnabled == true) {
        vtkSmartPointer<vtkDoubleArray> tArray;

        switch (fColorMode) {
            case sArea:
                tArray = fAreaData;
                break;
            case sAspect:
                tArray = fAspectData;
                break;
            case sModulo:
                tArray = fModuloData;
                break;
            default:
                tArray = fModuloData;
                break;
        }

        fMapper->SelectColorArray(tArray->GetName());
        fMapper->SetScalarRange(tArray->GetRange());
        fMapper->Update();

        vtkSmartPointer<vtkRenderer> vRenderer = fWindow->GetRenderer();
        vRenderer->AddActor(fActor);
    }
    return;
}
void KGVTKAxialMeshPainter::Write()
{
    if (fWriteEnabled == true) {
        string tFileName = string(OUTPUT_DEFAULT_DIR) + string("/") + GetName() + string(".vtp");

        vismsg(eNormal) << "vtk mesh painter <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells()
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

void KGVTKAxialMeshPainter::VisitSurface(KGSurface* aSurface)
{
    fCurrentOrigin = aSurface->GetOrigin();
    fCurrentXAxis = aSurface->GetXAxis();
    fCurrentYAxis = aSurface->GetYAxis();
    fCurrentZAxis = aSurface->GetZAxis();
    return;
}
void KGVTKAxialMeshPainter::VisitExtendedSurface(KGExtendedSurface<KGAxialMesh>* anExtendedSurface)
{
    fCurrentElements = anExtendedSurface->Elements();
    PaintElements();
    return;
}
void KGVTKAxialMeshPainter::VisitSpace(KGSpace* aSpace)
{
    fCurrentOrigin = aSpace->GetOrigin();
    fCurrentXAxis = aSpace->GetXAxis();
    fCurrentYAxis = aSpace->GetYAxis();
    fCurrentZAxis = aSpace->GetZAxis();
    return;
}
void KGVTKAxialMeshPainter::VisitExtendedSpace(KGExtendedSpace<KGAxialMesh>* anExtendedSpace)
{
    fCurrentElements = anExtendedSpace->Elements();
    PaintElements();
    return;
}

void KGVTKAxialMeshPainter::SetFile(const string& aFile)
{
    fFile = aFile;
    return;
}
const string& KGVTKAxialMeshPainter::GetFile() const
{
    return fFile;
}

void KGVTKAxialMeshPainter::SetArcCount(const unsigned int& aArcCount)
{
    fArcCount = aArcCount;
    return;
}
const unsigned int& KGVTKAxialMeshPainter::GetArcCount() const
{
    return fArcCount;
}

void KGVTKAxialMeshPainter::SetColorMode(const unsigned int& aColorMode)
{
    fColorMode = aColorMode;
    if (fColorMode > 2) {
        fColorMode = sModulo;
        return;
    }
    return;
}
const unsigned int& KGVTKAxialMeshPainter::GetColorMode() const
{
    return fColorMode;
}

void KGVTKAxialMeshPainter::PaintElements()
{
    vtkSmartPointer<vtkQuad> vQuad = vtkSmartPointer<vtkQuad>::New();
    vtkSmartPointer<vtkLine> vLine = vtkSmartPointer<vtkLine>::New();

    const unsigned int tModBase = 13;
    unsigned int tMod = 0;

    double tDeltaPhi = 2. * katrin::KConst::Pi() / (double) (fArcCount);
    double tThisPhi;
    double tNextPhi;
    KThreeVector tThisThisPoint;
    KThreeVector tThisNextPoint;
    KThreeVector tNextThisPoint;
    KThreeVector tNextNextPoint;

    for (auto* element : *fCurrentElements) {
        if (auto* tLoop = dynamic_cast<KGAxialMeshLoop*>(element)) {
            for (unsigned int tIndex = 0; tIndex < fArcCount; tIndex++) {
                tThisPhi = tIndex * tDeltaPhi;
                tNextPhi = (tIndex + 1) * tDeltaPhi;

                tThisThisPoint = fCurrentOrigin +
                                 tLoop->GetP0().Y() * (cos(tThisPhi) * fCurrentXAxis + sin(tThisPhi) * fCurrentYAxis) +
                                 tLoop->GetP0().X() * fCurrentZAxis;
                tThisNextPoint = fCurrentOrigin +
                                 tLoop->GetP0().Y() * (cos(tNextPhi) * fCurrentXAxis + sin(tNextPhi) * fCurrentYAxis) +
                                 tLoop->GetP0().X() * fCurrentZAxis;
                tNextThisPoint = fCurrentOrigin +
                                 tLoop->GetP1().Y() * (cos(tThisPhi) * fCurrentXAxis + sin(tThisPhi) * fCurrentYAxis) +
                                 tLoop->GetP1().X() * fCurrentZAxis;
                tNextNextPoint = fCurrentOrigin +
                                 tLoop->GetP1().Y() * (cos(tNextPhi) * fCurrentXAxis + sin(tNextPhi) * fCurrentYAxis) +
                                 tLoop->GetP1().X() * fCurrentZAxis;

                vtkIdType p0 = fPoints->InsertNextPoint(tThisThisPoint[0], tThisThisPoint[1], tThisThisPoint[2]);
                vtkIdType p1 = fPoints->InsertNextPoint(tThisNextPoint[0], tThisNextPoint[1], tThisNextPoint[2]);
                vtkIdType p2 = fPoints->InsertNextPoint(tNextNextPoint[0], tNextNextPoint[1], tNextNextPoint[2]);
                vtkIdType p3 = fPoints->InsertNextPoint(tNextThisPoint[0], tNextThisPoint[1], tNextThisPoint[2]);

                vQuad->GetPointIds()->SetId(0, p0);
                vQuad->GetPointIds()->SetId(1, p1);
                vQuad->GetPointIds()->SetId(2, p2);
                vQuad->GetPointIds()->SetId(3, p3);

                fPolyCells->InsertNextCell(vQuad);
                fAreaData->InsertNextTuple1(tLoop->Area());
                fAspectData->InsertNextTuple1(tLoop->Aspect());
                fModuloData->InsertNextTuple1(tMod % tModBase);
            }
            tMod++;
            continue;
        }

        if (auto* tRing = dynamic_cast<KGAxialMeshRing*>(element)) {
            for (unsigned int tIndex = 0; tIndex < fArcCount; tIndex++) {
                tThisPhi = tIndex * tDeltaPhi;
                tNextPhi = (tIndex + 1) * tDeltaPhi;

                tThisThisPoint = fCurrentOrigin +
                                 tRing->GetP0().Y() * (cos(tThisPhi) * fCurrentXAxis + sin(tThisPhi) * fCurrentYAxis) +
                                 tRing->GetP0().X() * fCurrentZAxis;
                tThisNextPoint = fCurrentOrigin +
                                 tRing->GetP0().Y() * (cos(tNextPhi) * fCurrentXAxis + sin(tNextPhi) * fCurrentYAxis) +
                                 tRing->GetP0().X() * fCurrentZAxis;

                vtkIdType p0 = fPoints->InsertNextPoint(tThisThisPoint[0], tThisThisPoint[1], tThisThisPoint[2]);
                vtkIdType p1 = fPoints->InsertNextPoint(tThisNextPoint[0], tThisNextPoint[1], tThisNextPoint[2]);

                vLine->GetPointIds()->SetId(0, p0);
                vLine->GetPointIds()->SetId(1, p1);

                fLineCells->InsertNextCell(vLine);
                fAreaData->InsertNextTuple1(tRing->Area());
                fAspectData->InsertNextTuple1(tRing->Aspect());
                fModuloData->InsertNextTuple1(tMod % tModBase);
            }
            tMod++;
            continue;
        }
    }

    return;
}
}  // namespace KGeoBag
