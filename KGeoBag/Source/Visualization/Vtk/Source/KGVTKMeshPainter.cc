#include "KGVTKMeshPainter.hh"

#include "KConst.h"
#include "KFile.h"
#include "KGMeshRectangle.hh"
#include "KGMeshTriangle.hh"
#include "KGMeshWire.hh"
#include "KGVisualizationMessage.hh"
#include "vtkCellData.h"
#include "vtkLine.h"
#include "vtkQuad.h"
#include "vtkTriangle.h"

#include <cmath>
#include <iostream>

using namespace std;

namespace KGeoBag
{
const unsigned int KGVTKMeshPainter::sArea = 0;
const unsigned int KGVTKMeshPainter::sAspect = 1;
const unsigned int KGVTKMeshPainter::sModulo = 2;

KGVTKMeshPainter::KGVTKMeshPainter() :
    fCurrentOrigin(KThreeVector::sZero),
    fCurrentXAxis(KThreeVector::sXUnit),
    fCurrentYAxis(KThreeVector::sYUnit),
    fCurrentZAxis(KThreeVector::sZUnit),
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
    fArcCount(6),
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
KGVTKMeshPainter::~KGVTKMeshPainter() = default;

void KGVTKMeshPainter::Render()
{
    return;
}
void KGVTKMeshPainter::Display()
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
void KGVTKMeshPainter::Write()
{
    if (fWriteEnabled == true) {
        string tFile;

        if (fFile.length() > 0) {
            tFile = string(OUTPUT_DEFAULT_DIR) + string("/") + fFile;
        }
        else {
            tFile = string(OUTPUT_DEFAULT_DIR) + string("/") + GetName() + string(".vtp");
        }

        vismsg(eNormal) << "vtk geometry painter <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells()
                        << "> cells to file <" << tFile << ">" << eom;

        vtkSmartPointer<vtkXMLPolyDataWriter> vWriter = fWindow->GetWriter();
        vWriter->SetFileName(tFile.c_str());
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

void KGVTKMeshPainter::VisitSurface(KGSurface* aSurface)
{
    fCurrentOrigin = aSurface->GetOrigin();
    fCurrentXAxis = aSurface->GetXAxis();
    fCurrentYAxis = aSurface->GetYAxis();
    fCurrentZAxis = aSurface->GetZAxis();
    return;
}
void KGVTKMeshPainter::VisitExtendedSurface(KGExtendedSurface<KGMesh>* anExtendedSurface)
{
    fCurrentElements = anExtendedSurface->Elements();
    PaintElements();
    return;
}
void KGVTKMeshPainter::VisitSpace(KGSpace* aSpace)
{
    fCurrentOrigin = aSpace->GetOrigin();
    fCurrentXAxis = aSpace->GetXAxis();
    fCurrentYAxis = aSpace->GetYAxis();
    fCurrentZAxis = aSpace->GetZAxis();
    return;
}
void KGVTKMeshPainter::VisitExtendedSpace(KGExtendedSpace<KGMesh>* anExtendedSpace)
{
    fCurrentElements = anExtendedSpace->Elements();
    PaintElements();
    return;
}

void KGVTKMeshPainter::SetFile(const string& aFile)
{
    fFile = aFile;
    return;
}
const string& KGVTKMeshPainter::GetFile() const
{
    return fFile;
}

void KGVTKMeshPainter::SetArcCount(const unsigned int& aArcCount)
{
    fArcCount = aArcCount;
    return;
}
const unsigned int& KGVTKMeshPainter::GetArcCount() const
{
    return fArcCount;
}

void KGVTKMeshPainter::SetColorMode(const unsigned int& aColorMode)
{
    fColorMode = aColorMode;
    if (fColorMode > 2) {
        fColorMode = sModulo;
        return;
    }
    return;
}
const unsigned int& KGVTKMeshPainter::GetColorMode() const
{
    return fColorMode;
}

void KGVTKMeshPainter::PaintElements()
{
    vtkSmartPointer<vtkTriangle> vTriangle = vtkSmartPointer<vtkTriangle>::New();
    vtkSmartPointer<vtkQuad> vQuad = vtkSmartPointer<vtkQuad>::New();

    unsigned int tMod = 0;
    const unsigned int tModBase = 13;

    unsigned int count = 0;
    for (auto* element : *fCurrentElements) {
        count++;

        if (auto* tMeshRectangle = dynamic_cast<KGMeshRectangle*>(element)) {
            KThreeVector tPoint0 = fCurrentOrigin + tMeshRectangle->GetP0().X() * fCurrentXAxis +
                                   tMeshRectangle->GetP0().Y() * fCurrentYAxis +
                                   tMeshRectangle->GetP0().Z() * fCurrentZAxis;
            KThreeVector tPoint1 = fCurrentOrigin + tMeshRectangle->GetP1().X() * fCurrentXAxis +
                                   tMeshRectangle->GetP1().Y() * fCurrentYAxis +
                                   tMeshRectangle->GetP1().Z() * fCurrentZAxis;
            KThreeVector tPoint2 = fCurrentOrigin + tMeshRectangle->GetP2().X() * fCurrentXAxis +
                                   tMeshRectangle->GetP2().Y() * fCurrentYAxis +
                                   tMeshRectangle->GetP2().Z() * fCurrentZAxis;
            KThreeVector tPoint3 = fCurrentOrigin + tMeshRectangle->GetP3().X() * fCurrentXAxis +
                                   tMeshRectangle->GetP3().Y() * fCurrentYAxis +
                                   tMeshRectangle->GetP3().Z() * fCurrentZAxis;

            vtkIdType vId0 = fPoints->InsertNextPoint(tPoint0.X(), tPoint0.Y(), tPoint0.Z());
            vtkIdType vId1 = fPoints->InsertNextPoint(tPoint1.X(), tPoint1.Y(), tPoint1.Z());
            vtkIdType vId2 = fPoints->InsertNextPoint(tPoint2.X(), tPoint2.Y(), tPoint2.Z());
            vtkIdType vId3 = fPoints->InsertNextPoint(tPoint3.X(), tPoint3.Y(), tPoint3.Z());

            vQuad->GetPointIds()->SetId(0, vId0);
            vQuad->GetPointIds()->SetId(1, vId1);
            vQuad->GetPointIds()->SetId(2, vId2);
            vQuad->GetPointIds()->SetId(3, vId3);

            fPolyCells->InsertNextCell(vQuad);
            fAreaData->InsertNextTuple1(tMeshRectangle->Area());
            fAspectData->InsertNextTuple1(tMeshRectangle->Aspect());
            fModuloData->InsertNextTuple1(tMod % tModBase);

            tMod++;
        }

        if (auto* tMeshTriangle = dynamic_cast<KGMeshTriangle*>(element)) {
            KThreeVector tPoint0 = fCurrentOrigin + tMeshTriangle->GetP0().X() * fCurrentXAxis +
                                   tMeshTriangle->GetP0().Y() * fCurrentYAxis +
                                   tMeshTriangle->GetP0().Z() * fCurrentZAxis;
            KThreeVector tPoint1 = fCurrentOrigin + tMeshTriangle->GetP1().X() * fCurrentXAxis +
                                   tMeshTriangle->GetP1().Y() * fCurrentYAxis +
                                   tMeshTriangle->GetP1().Z() * fCurrentZAxis;
            KThreeVector tPoint2 = fCurrentOrigin + tMeshTriangle->GetP2().X() * fCurrentXAxis +
                                   tMeshTriangle->GetP2().Y() * fCurrentYAxis +
                                   tMeshTriangle->GetP2().Z() * fCurrentZAxis;

            vtkIdType vId0 = fPoints->InsertNextPoint(tPoint0.X(), tPoint0.Y(), tPoint0.Z());
            vtkIdType vId1 = fPoints->InsertNextPoint(tPoint1.X(), tPoint1.Y(), tPoint1.Z());
            vtkIdType vId2 = fPoints->InsertNextPoint(tPoint2.X(), tPoint2.Y(), tPoint2.Z());

            vTriangle->GetPointIds()->SetId(0, vId0);
            vTriangle->GetPointIds()->SetId(1, vId1);
            vTriangle->GetPointIds()->SetId(2, vId2);

            fPolyCells->InsertNextCell(vTriangle);
            fAreaData->InsertNextTuple1(tMeshTriangle->Area());
            fAspectData->InsertNextTuple1(tMeshTriangle->Aspect());
            fModuloData->InsertNextTuple1(tMod % tModBase);

            tMod++;
        }

        if (auto* tMeshWire = dynamic_cast<KGMeshWire*>(element)) {
            KThreeVector tStart = fCurrentOrigin + tMeshWire->GetP1().X() * fCurrentXAxis +
                                  tMeshWire->GetP1().Y() * fCurrentYAxis + tMeshWire->GetP1().Z() * fCurrentZAxis;
            KThreeVector tEnd = fCurrentOrigin + tMeshWire->GetP0().X() * fCurrentXAxis +
                                tMeshWire->GetP0().Y() * fCurrentYAxis + tMeshWire->GetP0().Z() * fCurrentZAxis;
            KThreeVector tConnection = tEnd - tStart;
            KThreeVector tOrthogonal = tConnection.Orthogonal();
            KThreeVector tU = tOrthogonal.Unit();
            KThreeVector tV = tConnection.Cross(tOrthogonal).Unit();
            KThreeVector tUnitPoint;
            KThreeVector tStartPoint;
            KThreeVector tEndPoint;

            double tRadius = tMeshWire->GetDiameter() / 2.;
            double tIncrement = (2. * katrin::KConst::Pi()) / ((double) (fArcCount));
            double tPhi;

            vector<vtkIdType> tStartIds;
            vector<vtkIdType> tEndIds;

            for (unsigned int tIndex = 0; tIndex < fArcCount; tIndex++) {
                tPhi = tIndex * tIncrement;
                tUnitPoint = tRadius * (cos(tPhi) * tU + sin(tPhi) * tV);

                tStartPoint = tStart + tUnitPoint;
                tStartIds.push_back(fPoints->InsertNextPoint(tStartPoint.X(), tStartPoint.Y(), tStartPoint.Z()));

                tEndPoint = tEnd + tUnitPoint;
                tEndIds.push_back(fPoints->InsertNextPoint(tEndPoint.X(), tEndPoint.Y(), tEndPoint.Z()));
            }

            for (unsigned int tIndex = 0; tIndex < fArcCount - 1; tIndex++) {
                vQuad->GetPointIds()->SetId(0, tStartIds.at(tIndex));
                vQuad->GetPointIds()->SetId(1, tStartIds.at(tIndex + 1));
                vQuad->GetPointIds()->SetId(2, tEndIds.at(tIndex + 1));
                vQuad->GetPointIds()->SetId(3, tEndIds.at(tIndex));

                fPolyCells->InsertNextCell(vQuad);
                fAreaData->InsertNextTuple1(tMeshWire->Area());
                fAspectData->InsertNextTuple1(tMeshWire->Aspect());
                fModuloData->InsertNextTuple1(tMod % tModBase);
            }

            vQuad->GetPointIds()->SetId(0, tStartIds.at(fArcCount - 1));
            vQuad->GetPointIds()->SetId(1, tStartIds.at(0));
            vQuad->GetPointIds()->SetId(2, tEndIds.at(0));
            vQuad->GetPointIds()->SetId(3, tEndIds.at(fArcCount - 1));

            fPolyCells->InsertNextCell(vQuad);
            fAreaData->InsertNextTuple1(tMeshWire->Area());
            fAspectData->InsertNextTuple1(tMeshWire->Aspect());
            fModuloData->InsertNextTuple1(tMod % tModBase);

            tMod++;
        }
    }

    return;
}
}  // namespace KGeoBag
