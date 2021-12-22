#include "KGVTKGeometryPainter.hh"

#include "KFile.h"
#include "KGVisualizationMessage.hh"

#ifdef KGeoBag_USE_BOOST
//#include "KPathUtils.h"
//using katrin::KPathUtils;
#endif

#include "KConst.h"
#include "vtkAppendPolyData.h"
#include "vtkCellArray.h"
#include "vtkCutter.h"
#include "vtkDepthSortPolyData.h"
#include "vtkIVWriter.h"
#include "vtkPolyDataMapper.h"
#include "vtkTriangle.h"
#include "vtkQuad.h"
#include "vtkPlane.h"
#include "vtkPolygon.h"
#include "vtkSTLWriter.h"
#include "vtkTriangleFilter.h"
#include "vtkXMLPolyDataWriter.h"
#include "vtkUnsignedCharArray.h"

#include <cmath>

using namespace std;

using katrin::KTwoVector;
using katrin::KThreeVector;
using katrin::KTransformation;

namespace KGeoBag
{

KGVTKGeometryPainter::KGVTKGeometryPainter() :
    fPath(""),
    fWriteSTL(false),
    fPlaneMode(0),
    fPoints(vtkSmartPointer<vtkPoints>::New()),
    fCells(vtkSmartPointer<vtkCellArray>::New()),
    fColors(vtkSmartPointer<vtkUnsignedCharArray>::New()),
    fPolyData(vtkSmartPointer<vtkPolyData>::New()),
    fMapper(vtkSmartPointer<vtkPolyDataMapper>::New()),
    fActor(vtkSmartPointer<vtkActor>::New()),
    fCurrentSpace(nullptr),
    fCurrentSurface(nullptr),
    fCurrentData(nullptr),
    fCurrentOrigin(KThreeVector::sZero),
    fCurrentXAxis(KThreeVector::sXUnit),
    fCurrentYAxis(KThreeVector::sYUnit),
    fCurrentZAxis(KThreeVector::sZUnit),
    fIgnore(true)
{
    fColors->SetNumberOfComponents(4);
    fPolyData->SetPoints(fPoints);
    fPolyData->SetPolys(fCells);
    fPolyData->GetCellData()->SetScalars(fColors);
#ifdef VTK6
    fMapper->SetInputData(fPolyData);
#else
    fMapper->SetInput(fPolyData);
#endif
    fMapper->SetScalarModeToUseCellData();
    fActor->SetMapper(fMapper);
    fDefaultData.SetColor(KGRGBAColor(255, 255, 255, 100));
    fDefaultData.SetArc(72);
}
KGVTKGeometryPainter::~KGVTKGeometryPainter() = default;

void KGVTKGeometryPainter::Render()
{
    KGSurface* tSurface;
    vector<KGSurface*>::iterator tSurfaceIt;
    for (tSurfaceIt = fSurfaces.begin(); tSurfaceIt != fSurfaces.end(); tSurfaceIt++) {
        tSurface = *tSurfaceIt;
        tSurface->AcceptNode(this);
    }

    KGSpace* tSpace;
    vector<KGSpace*>::iterator tSpaceIt;
    for (tSpaceIt = fSpaces.begin(); tSpaceIt != fSpaces.end(); tSpaceIt++) {
        tSpace = *tSpaceIt;
        tSpace->AcceptNode(this);
    }

    return;
}
void KGVTKGeometryPainter::Display()
{
    if (fDisplayEnabled == true) {
        vtkSmartPointer<vtkRenderer> vRenderer = fWindow->GetRenderer();
        vRenderer->AddActor(fActor);
    }
    return;
}
void KGVTKGeometryPainter::Write()
{
    if (fWriteEnabled == true) {
        WriteVTK();

        if (fWriteSTL == true)
            WriteSTL();
    }
}
void KGVTKGeometryPainter::WriteVTK()
{
    string tFile;

    if (!fFile.empty()) {
        if (fPath.empty()) {
            tFile = string(OUTPUT_DEFAULT_DIR) + string("/") + fFile;
        }
        else {
#ifdef KGeoBag_USE_BOOST
//                KPathUtils::MakeDirectory( fPath );
#endif
            tFile = fPath + string("/") + fFile;
        }
    }
    else {
        tFile = string(OUTPUT_DEFAULT_DIR) + string("/") + GetName() + string(".vtp");
    }

    vismsg(eNormal) << "vtk geometry painter <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells()
                    << "> cells to file <" << tFile << ">" << eom;

    vtkSmartPointer<vtkXMLPolyDataWriter> vWriter = fWindow->GetWriter();
#ifdef VTK6
    vWriter->SetInputData(fPolyData);
#else
    vWriter->SetInput(fPolyData);
#endif
    vWriter->SetFileName(tFile.c_str());
    vWriter->SetDataModeToBinary();
    vWriter->Write();
    return;
}
void KGVTKGeometryPainter::WriteSTL()
{
    string tFile;

    if (!fFile.empty()) {
        if (fPath.empty()) {
            tFile = string(OUTPUT_DEFAULT_DIR) + string("/") + fFile;
        }
        else {
#ifdef KGeoBag_USE_BOOST
//                KPathUtils::MakeDirectory( fPath );
#endif
            tFile = fPath + string("/") + fFile;
        }

        // change extension
        int tIndex = tFile.find_last_of('.');
        tFile = tFile.substr(0, tIndex) + string(".stl");
    }
    else {
        tFile = string(OUTPUT_DEFAULT_DIR) + string("/") + GetName() + string(".stl");
    }

    vismsg(eNormal) << "vtk geometry painter <" << GetName() << "> is writing <" << fPolyData->GetNumberOfCells()
                    << "> cells to STL file <" << tFile << ">" << eom;

    // use triangle filter because STL can only export 3 vertices per object (see vtkSTLWriter documentation)
    vtkSmartPointer<vtkTriangleFilter> vTriangleFilter = vtkSmartPointer<vtkTriangleFilter>::New();
#ifdef VTK6
    vTriangleFilter->SetInputData(fPolyData);
#else
    vTriangleFilter->SetInput(fPolyData);
#endif

    vtkSmartPointer<vtkSTLWriter> vWriter = vtkSmartPointer<vtkSTLWriter>::New();
    vWriter->SetInputConnection(vTriangleFilter->GetOutputPort());
    vWriter->SetFileName(tFile.c_str());
    vWriter->SetFileTypeToASCII();  // binary STL might break import in other programs
    vWriter->Write();
    return;
}

void KGVTKGeometryPainter::SetFile(const string& aFile)
{
    fFile = aFile;
    return;
}
const string& KGVTKGeometryPainter::GetFile() const
{
    return fFile;
}
void KGVTKGeometryPainter::SetPath(const string& aPath)
{
    fPath = aPath;
    return;
}
void KGVTKGeometryPainter::SetWriteSTL(bool aFlag)
{
    fWriteSTL = aFlag;
    return;
}

void KGVTKGeometryPainter::AddSurface(KGSurface* aSurface)
{
    fSurfaces.push_back(aSurface);
    return;
}
void KGVTKGeometryPainter::AddSpace(KGSpace* aSpace)
{
    fSpaces.push_back(aSpace);
    return;
}

string KGVTKGeometryPainter::HelpText()
{
    std::ostringstream tText;

    tText << "  geometry painter:" << '\n';
    tText << "    X,Y,Z - toggle slice mode (cut through plane with x/y/z normal)" << '\n';

    return tText.str();
}

void KGVTKGeometryPainter::OnKeyPress(vtkObject* aCaller, long unsigned int /*eventId*/, void* /*aClient*/, void* /*callData*/)
{
    auto* tInteractor = static_cast<vtkRenderWindowInteractor*>(aCaller);

    string Symbol = tInteractor->GetKeySym();
    bool WithShift = tInteractor->GetShiftKey();
    bool WithCtrl = tInteractor->GetControlKey();

    //utilmsg(eDebug) << "key press in VTK geometry painter: " << Symbol << (WithShift ? "+shift" : "") << (WithCtrl ? "+ctrl" : "") << eom;

    if ((WithShift == false) && (WithCtrl == false)) {
        //screenshot
        if (Symbol == string("x")) {
            if (fPlaneMode != 1) {
                vtkSmartPointer<vtkPlane> vPlane = vtkSmartPointer<vtkPlane>::New();
                vPlane->SetOrigin(0., 0., 0.);
                vPlane->SetNormal(1., 0., 0.);

                vtkSmartPointer<vtkCutter> vCutter = vtkSmartPointer<vtkCutter>::New();
                vCutter->SetCutFunction(vPlane);

                #ifdef VTK6
                    vCutter->SetInputData(fPolyData);
                    fMapper->SetInputConnection(vCutter->GetOutputPort());
                #else
                    vCutter->SetInput(fPolyData);
                    fMapper->SetInput(vCutter);
                #endif

                vCutter->Update();
                fMapper->Update();
                fPlaneMode = 1;
            }
            else {
                #ifdef VTK6
                    fMapper->SetInputData(fPolyData);
                #else
                    fMapper->SetInput(fPolyData);
                #endif

                fMapper->Update();
                fPlaneMode = 0;
            }
        }
        else if (Symbol == string("y")) {
            if (fPlaneMode != 2) {
                vtkSmartPointer<vtkPlane> vPlane = vtkSmartPointer<vtkPlane>::New();
                vPlane->SetOrigin(0., 0., 0.);
                vPlane->SetNormal(0., 1., 0.);

                vtkSmartPointer<vtkCutter> vCutter = vtkSmartPointer<vtkCutter>::New();
                vCutter->SetCutFunction(vPlane);

                #ifdef VTK6
                    vCutter->SetInputData(fPolyData);
                    fMapper->SetInputConnection(vCutter->GetOutputPort());
                #else
                    vCutter->SetInput(fPolyData);
                    fMapper->SetInput(vCutter);
                #endif

                vCutter->Update();
                fMapper->Update();
                fPlaneMode = 2;
            }
            else {
                #ifdef VTK6
                    fMapper->SetInputData(fPolyData);
                #else
                    fMapper->SetInput(fPolyData);
                #endif

                fMapper->Update();
                fPlaneMode = 0;
            }
        }
        else if (Symbol == string("z")) {
            if (fPlaneMode != 3) {
                vtkSmartPointer<vtkPlane> vPlane = vtkSmartPointer<vtkPlane>::New();
                vPlane->SetOrigin(0., 0., 0.);
                vPlane->SetNormal(0., 0., 1.);

                vtkSmartPointer<vtkCutter> vCutter = vtkSmartPointer<vtkCutter>::New();
                vCutter->SetCutFunction(vPlane);

                #ifdef VTK6
                    vCutter->SetInputData(fPolyData);
                    fMapper->SetInputConnection(vCutter->GetOutputPort());
                #else
                    vCutter->SetInput(fPolyData);
                    fMapper->SetInput(vCutter);
                #endif

                vCutter->Update();
                fMapper->Update();
                fPlaneMode = 3;
            }
            else {
                #ifdef VTK6
                    fMapper->SetInputData(fPolyData);
                #else
                    fMapper->SetInput(fPolyData);
                #endif

                fMapper->Update();
                fPlaneMode = 0;
            }
        }
    }
}

//****************
//surface visitors
//****************

void KGVTKGeometryPainter::VisitSurface(KGSurface* aSurface)
{
    fCurrentSurface = aSurface;
    fCurrentOrigin = aSurface->GetOrigin();
    fCurrentXAxis = aSurface->GetXAxis();
    fCurrentYAxis = aSurface->GetYAxis();
    fCurrentZAxis = aSurface->GetZAxis();

    if (aSurface->HasExtension<KGAppearance>() == true) {
        fCurrentData = aSurface->AsExtension<KGAppearance>();
    }
    else {
        fCurrentData = &fDefaultData;
    }

    if (fCurrentSpace != nullptr) {
        for (auto* tIt : *fCurrentSpace->GetBoundaries()) {
            if (tIt == fCurrentSurface) {
                if (fCurrentData == &fDefaultData) {
                    fIgnore = true;
                }
                else {
                    fIgnore = false;
                }
            }
        }
    }
    else {
        fIgnore = false;
    }

    return;
}
void KGVTKGeometryPainter::VisitFlattenedClosedPathSurface(KGFlattenedCircleSurface* aFlattenedCircleSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create circle points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aFlattenedCircleSurface->Path().operator->(), tCirclePoints);

    //create flattened points
    KThreeVector tApexPoint;
    TubeMesh tMeshPoints;
    ClosedPointsFlattenedToTubeMeshAndApex(tCirclePoints,
                                           aFlattenedCircleSurface->Path()->Centroid(),
                                           aFlattenedCircleSurface->Z(),
                                           tMeshPoints,
                                           tApexPoint);

    //create mesh
    TubeMeshToVTK(tMeshPoints, tApexPoint);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitFlattenedClosedPathSurface(KGFlattenedPolyLoopSurface* aFlattenedPolyLoopSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create circle points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aFlattenedPolyLoopSurface->Path().operator->(), tPolyLoopPoints);

    //create flattened points
    KThreeVector tApexPoint;
    TubeMesh tMeshPoints;
    ClosedPointsFlattenedToTubeMeshAndApex(tPolyLoopPoints,
                                           aFlattenedPolyLoopSurface->Path()->Centroid(),
                                           aFlattenedPolyLoopSurface->Z(),
                                           tMeshPoints,
                                           tApexPoint);

    //create mesh
    TubeMeshToVTK(tMeshPoints, tApexPoint);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitRotatedPathSurface(KGRotatedLineSegmentSurface* aRotatedLineSegmentSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create line segment points
    OpenPoints tLineSegmentPoints;
    LineSegmentToOpenPoints(aRotatedLineSegmentSurface->Path().operator->(), tLineSegmentPoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tLineSegmentPoints, tMeshPoints);

    //surgery
    bool tHasStart = false;
    KThreeVector tStartApex;
    if (aRotatedLineSegmentSurface->Path()->Start().Y() == 0.) {
        tHasStart = true;
        tStartApex.SetComponents(0., 0., aRotatedLineSegmentSurface->Path()->Start().X());
        tMeshPoints.fData.pop_front();
    }

    bool tHasEnd = false;
    KThreeVector tEndApex;
    if (aRotatedLineSegmentSurface->Path()->End().Y() == 0.) {
        tHasEnd = true;
        tEndApex.SetComponents(0., 0., aRotatedLineSegmentSurface->Path()->End().X());
        tMeshPoints.fData.pop_back();
    }

    //create mesh
    if (tHasStart == true) {
        if (tHasEnd == true) {
            TubeMeshToVTK(tStartApex, tMeshPoints, tEndApex);
        }
        else {
            TubeMeshToVTK(tStartApex, tMeshPoints);
        }
    }
    else {
        if (tHasEnd == true) {
            TubeMeshToVTK(tMeshPoints, tEndApex);
        }
        else {
            TubeMeshToVTK(tMeshPoints);
        }
    }

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitRotatedPathSurface(KGRotatedArcSegmentSurface* aRotatedArcSegmentSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create arc segment points
    OpenPoints tArcSegmentPoints;
    ArcSegmentToOpenPoints(aRotatedArcSegmentSurface->Path().operator->(), tArcSegmentPoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tArcSegmentPoints, tMeshPoints);

    //surgery
    bool tHasStart = false;
    KThreeVector tStartApex;
    if (aRotatedArcSegmentSurface->Path()->Start().Y() == 0.) {
        tHasStart = true;
        tStartApex.SetComponents(0., 0., aRotatedArcSegmentSurface->Path()->Start().X());
        tMeshPoints.fData.pop_front();
    }

    bool tHasEnd = false;
    KThreeVector tEndApex;
    if (aRotatedArcSegmentSurface->Path()->End().Y() == 0.) {
        tHasEnd = true;
        tEndApex.SetComponents(0., 0., aRotatedArcSegmentSurface->Path()->End().X());
        tMeshPoints.fData.pop_back();
    }

    //create mesh
    if (tHasStart == true) {
        if (tHasEnd == true) {
            TubeMeshToVTK(tStartApex, tMeshPoints, tEndApex);
        }
        else {
            TubeMeshToVTK(tStartApex, tMeshPoints);
        }
    }
    else {
        if (tHasEnd == true) {
            TubeMeshToVTK(tMeshPoints, tEndApex);
        }
        else {
            TubeMeshToVTK(tMeshPoints);
        }
    }

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitRotatedPathSurface(KGRotatedPolyLineSurface* aRotatedPolyLineSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create poly line points
    OpenPoints tPolyLinePoints;
    PolyLineToOpenPoints(aRotatedPolyLineSurface->Path().operator->(), tPolyLinePoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tPolyLinePoints, tMeshPoints);

    //surgery
    bool tHasStart = false;
    KThreeVector tStartApex;
    if (aRotatedPolyLineSurface->Path()->Start().Y() == 0.) {
        tHasStart = true;
        tStartApex.SetComponents(0., 0., aRotatedPolyLineSurface->Path()->Start().X());
        tMeshPoints.fData.pop_front();
    }

    bool tHasEnd = false;
    KThreeVector tEndApex;
    if (aRotatedPolyLineSurface->Path()->End().Y() == 0.) {
        tHasEnd = true;
        tEndApex.SetComponents(0., 0., aRotatedPolyLineSurface->Path()->End().X());
        tMeshPoints.fData.pop_back();
    }

    //create mesh
    if (tHasStart == true) {
        if (tHasEnd == true) {
            TubeMeshToVTK(tStartApex, tMeshPoints, tEndApex);
        }
        else {
            TubeMeshToVTK(tStartApex, tMeshPoints);
        }
    }
    else {
        if (tHasEnd == true) {
            TubeMeshToVTK(tMeshPoints, tEndApex);
        }
        else {
            TubeMeshToVTK(tMeshPoints);
        }
    }

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitRotatedPathSurface(KGRotatedCircleSurface* aRotatedCircleSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create poly line points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aRotatedCircleSurface->Path().operator->(), tCirclePoints);

    //create rotated points
    TorusMesh tMeshPoints;
    ClosedPointsRotatedToTorusMesh(tCirclePoints, tMeshPoints);

    //create mesh
    TorusMeshToVTK(tMeshPoints);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitRotatedPathSurface(KGRotatedPolyLoopSurface* aRotatedPolyLoopSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create poly loop points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aRotatedPolyLoopSurface->Path().operator->(), tPolyLoopPoints);

    //create rotated points
    TorusMesh tMeshPoints;
    ClosedPointsRotatedToTorusMesh(tPolyLoopPoints, tMeshPoints);

    //create mesh
    TorusMeshToVTK(tMeshPoints);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitShellPathSurface(KGShellLineSegmentSurface* aShellLineSegmentSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create line segment points
    OpenPoints tLineSegmentPoints;
    LineSegmentToOpenPoints(aShellLineSegmentSurface->Path().operator->(), tLineSegmentPoints);

    //create rotated points
    ShellMesh tMeshPoints;
    OpenPointsRotatedToShellMesh(tLineSegmentPoints,
                                 tMeshPoints,
                                 aShellLineSegmentSurface->AngleStart(),
                                 aShellLineSegmentSurface->AngleStop());


    ShellMeshToVTK(tMeshPoints);
    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitShellPathSurface(KGShellArcSegmentSurface* aShellArcSegmentSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create line segment points
    OpenPoints tArcSegmentPoints;
    ArcSegmentToOpenPoints(aShellArcSegmentSurface->Path().operator->(), tArcSegmentPoints);

    //create rotated points
    ShellMesh tMeshPoints;
    OpenPointsRotatedToShellMesh(tArcSegmentPoints,
                                 tMeshPoints,
                                 aShellArcSegmentSurface->AngleStart(),
                                 aShellArcSegmentSurface->AngleStop());


    ShellMeshToVTK(tMeshPoints);
    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitShellPathSurface(KGShellPolyLineSurface* aShellPolyLineSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create poly line points
    OpenPoints tPolyLinePoints;
    PolyLineToOpenPoints(aShellPolyLineSurface->Path().operator->(), tPolyLinePoints);

    //create shell points
    ShellMesh tMeshPoints;
    OpenPointsRotatedToShellMesh(tPolyLinePoints,
                                 tMeshPoints,
                                 aShellPolyLineSurface->AngleStart(),
                                 aShellPolyLineSurface->AngleStop());


    //create mesh

    ShellMeshToVTK(tMeshPoints);


    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitShellPathSurface(KGShellPolyLoopSurface* aShellPolyLoopSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create poly loop points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aShellPolyLoopSurface->Path().operator->(), tPolyLoopPoints);

    //create rotated points
    ShellMesh tMeshPoints;
    ClosedPointsRotatedToShellMesh(tPolyLoopPoints,
                                   tMeshPoints,
                                   aShellPolyLoopSurface->AngleStart(),
                                   aShellPolyLoopSurface->AngleStop());

    //create mesh
    ClosedShellMeshToVTK(tMeshPoints);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitShellPathSurface(KGShellCircleSurface* aShellCircleSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create poly line points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aShellCircleSurface->Path().operator->(), tCirclePoints);

    //create rotated points
    ShellMesh tMeshPoints;
    ClosedPointsRotatedToShellMesh(tCirclePoints,
                                   tMeshPoints,
                                   aShellCircleSurface->AngleStart(),
                                   aShellCircleSurface->AngleStop());

    //create mesh
    ClosedShellMeshToVTK(tMeshPoints);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitExtrudedPathSurface(KGExtrudedLineSegmentSurface* aExtrudedLineSegmentSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create line segment points
    OpenPoints tLineSegmentPoints;
    LineSegmentToOpenPoints(aExtrudedLineSegmentSurface->Path().operator->(), tLineSegmentPoints);

    //create extruded points
    FlatMesh tMeshPoints;
    OpenPointsExtrudedToFlatMesh(tLineSegmentPoints,
                                 aExtrudedLineSegmentSurface->ZMin(),
                                 aExtrudedLineSegmentSurface->ZMax(),
                                 tMeshPoints);

    //create mesh
    FlatMeshToVTK(tMeshPoints);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitExtrudedPathSurface(KGExtrudedArcSegmentSurface* aExtrudedArcSegmentSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create arc segment points
    OpenPoints tArcSegmentPoints;
    ArcSegmentToOpenPoints(aExtrudedArcSegmentSurface->Path().operator->(), tArcSegmentPoints);

    //create extruded points
    FlatMesh tMeshPoints;
    OpenPointsExtrudedToFlatMesh(tArcSegmentPoints,
                                 aExtrudedArcSegmentSurface->ZMin(),
                                 aExtrudedArcSegmentSurface->ZMax(),
                                 tMeshPoints);

    //create mesh
    FlatMeshToVTK(tMeshPoints);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitExtrudedPathSurface(KGExtrudedPolyLineSurface* aExtrudedPolyLineSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create poly line points
    OpenPoints tPolyLinePoints;
    PolyLineToOpenPoints(aExtrudedPolyLineSurface->Path().operator->(), tPolyLinePoints);

    //create extruded points
    FlatMesh tMeshPoints;
    OpenPointsExtrudedToFlatMesh(tPolyLinePoints,
                                 aExtrudedPolyLineSurface->ZMin(),
                                 aExtrudedPolyLineSurface->ZMax(),
                                 tMeshPoints);

    //create mesh
    FlatMeshToVTK(tMeshPoints);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitExtrudedPathSurface(KGExtrudedCircleSurface* aExtrudedCircleSurface)
{
    if (fIgnore == true) {
        return;
    }
    //create poly line points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aExtrudedCircleSurface->Path().operator->(), tCirclePoints);

    //create rotated points
    TubeMesh tMeshPoints;
    ClosedPointsExtrudedToTubeMesh(tCirclePoints,
                                   aExtrudedCircleSurface->ZMin(),
                                   aExtrudedCircleSurface->ZMax(),
                                   tMeshPoints);

    //create mesh
    TubeMeshToVTK(tMeshPoints);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitExtrudedPathSurface(KGExtrudedPolyLoopSurface* aExtrudedPolyLoopSurface)
{
    if (fIgnore == true) {
        return;
    }

    //create poly loop points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aExtrudedPolyLoopSurface->Path().operator->(), tPolyLoopPoints);

    //create rotated points
    TubeMesh tMeshPoints;
    ClosedPointsExtrudedToTubeMesh(tPolyLoopPoints,
                                   aExtrudedPolyLoopSurface->ZMin(),
                                   aExtrudedPolyLoopSurface->ZMax(),
                                   tMeshPoints);

    //create mesh
    TubeMeshToVTK(tMeshPoints);

    //clear surface
    fCurrentSurface = nullptr;

    return;
}

void KGVTKGeometryPainter::VisitWrappedSurface(KGConicalWireArraySurface* aConicalWireArraySurface)
{
    if (fIgnore == true) {
        return;
    }

    // get the wire radius
    double tWireRadius = (aConicalWireArraySurface->GetObject()->GetDiameter()) / 2.;

    // create three-dimensional poly line points
    ThreePoints tWireArrayThreePoints;
    WireArrayToThreePoints(aConicalWireArraySurface, tWireArrayThreePoints);

    // create rotated points and create mesh
    TubeMesh tMeshPoints;
    ThreePointsToTubeMeshToVTK(tWireArrayThreePoints, tMeshPoints, tWireRadius);

    //clear surface
    fCurrentSurface = nullptr;
}

void KGVTKGeometryPainter::VisitWrappedSurface(KGRodSurface* aRodSurface)
{
    if (fIgnore == true) {
        return;
    }

    // getting the radius of the rod elements
    double tRodRadius = aRodSurface->GetObject()->GetRadius();

    // create three-dimensional poly line points
    ThreePoints tRodThreePoints;
    RodsToThreePoints(aRodSurface, tRodThreePoints);

    //create rotated points
    TubeMesh tMeshPoints;
    ThreePointsToTubeMeshToVTK(tRodThreePoints, tMeshPoints, tRodRadius);

    //clear space
    fCurrentSurface = nullptr;

    return;
}

void KGVTKGeometryPainter::VisitWrappedSurface(KGStlFileSurface* aStlFileSurface)
{
    if (fIgnore == true) {
        return;
    }

    // create rotated points and create mesh
    Mesh tMeshPoints;
    for (auto & elem : aStlFileSurface->GetObject()->GetElements()) {
        Mesh::Group tMeshGroup = { elem.GetP0(), elem.GetP1(), elem.GetP2() };
        tMeshPoints.fData.push_back(tMeshGroup);
    }

    //create mesh
    MeshToVTK(tMeshPoints);

    //clear space
    fCurrentSurface = nullptr;
}

//**************
//space visitors
//**************

void KGVTKGeometryPainter::VisitSpace(KGSpace* aSpace)
{
    fCurrentSpace = aSpace;
    fCurrentOrigin = aSpace->GetOrigin();
    fCurrentXAxis = aSpace->GetXAxis();
    fCurrentYAxis = aSpace->GetYAxis();
    fCurrentZAxis = aSpace->GetZAxis();

    if (aSpace->HasExtension<KGAppearance>() == true) {
        fCurrentData = aSpace->AsExtension<KGAppearance>();
    }
    else {
        fCurrentData = &fDefaultData;
    }

    return;
}
void KGVTKGeometryPainter::VisitRotatedOpenPathSpace(KGRotatedLineSegmentSpace* aRotatedLineSegmentSpace)
{
    //create line segment points
    OpenPoints tLineSegmentPoints;
    LineSegmentToOpenPoints(aRotatedLineSegmentSpace->Path().operator->(), tLineSegmentPoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tLineSegmentPoints, tMeshPoints);

    //create start circle points
    ClosedPoints tStartCirclePoints;
    CircleToClosedPoints(aRotatedLineSegmentSpace->StartPath().operator->(), tStartCirclePoints);

    //create start flattened mesh points
    TubeMesh tStartMeshPoints;
    KThreeVector tStartApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tStartCirclePoints,
                                           aRotatedLineSegmentSpace->StartPath()->Centroid(),
                                           aRotatedLineSegmentSpace->Path()->Start().X(),
                                           tStartMeshPoints,
                                           tStartApex);

    //create end circle points
    ClosedPoints tEndCirclePoints;
    CircleToClosedPoints(aRotatedLineSegmentSpace->EndPath().operator->(), tEndCirclePoints);

    //create end flattened mesh points
    TubeMesh tEndMeshPoints;
    KThreeVector tEndApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tEndCirclePoints,
                                           aRotatedLineSegmentSpace->EndPath()->Centroid(),
                                           aRotatedLineSegmentSpace->Path()->End().X(),
                                           tEndMeshPoints,
                                           tEndApex);

    //surgery
    if (aRotatedLineSegmentSpace->Path()->Start().Y() > 0) {
        auto tCircleIt = ++(tStartMeshPoints.fData.begin());
        while (tCircleIt != tStartMeshPoints.fData.end()) {
            tMeshPoints.fData.push_front(*tCircleIt);
            ++tCircleIt;
        }
    }
    else {
        tMeshPoints.fData.pop_front();
    }

    if (aRotatedLineSegmentSpace->Path()->End().Y() > 0) {
        auto tCircleIt = ++(tEndMeshPoints.fData.begin());
        while (tCircleIt != tEndMeshPoints.fData.end()) {
            tMeshPoints.fData.push_back(*tCircleIt);
            ++tCircleIt;
        }
    }
    else {
        tMeshPoints.fData.pop_back();
    }

    TubeMeshToVTK(tStartApex, tMeshPoints, tEndApex);

    //clear space
    fCurrentSpace = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitRotatedOpenPathSpace(KGRotatedArcSegmentSpace* aRotatedArcSegmentSpace)
{
    //create line segment points
    OpenPoints tArcSegmentPoints;
    ArcSegmentToOpenPoints(aRotatedArcSegmentSpace->Path().operator->(), tArcSegmentPoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tArcSegmentPoints, tMeshPoints);

    //create start circle points
    ClosedPoints tStartCirclePoints;
    CircleToClosedPoints(aRotatedArcSegmentSpace->StartPath().operator->(), tStartCirclePoints);

    //create start flattened mesh points
    TubeMesh tStartMeshPoints;
    KThreeVector tStartApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tStartCirclePoints,
                                           aRotatedArcSegmentSpace->StartPath()->Centroid(),
                                           aRotatedArcSegmentSpace->Path()->Start().X(),
                                           tStartMeshPoints,
                                           tStartApex);

    //create end circle points
    ClosedPoints tEndCirclePoints;
    CircleToClosedPoints(aRotatedArcSegmentSpace->EndPath().operator->(), tEndCirclePoints);

    //create end flattened mesh points
    TubeMesh tEndMeshPoints;
    KThreeVector tEndApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tEndCirclePoints,
                                           aRotatedArcSegmentSpace->EndPath()->Centroid(),
                                           aRotatedArcSegmentSpace->Path()->End().X(),
                                           tEndMeshPoints,
                                           tEndApex);

    //surgery
    if (aRotatedArcSegmentSpace->Path()->Start().Y() > 0) {
        auto tCircleIt = ++(tStartMeshPoints.fData.begin());
        while (tCircleIt != tStartMeshPoints.fData.end()) {
            tMeshPoints.fData.push_front(*tCircleIt);
            ++tCircleIt;
        }
    }
    else {
        tMeshPoints.fData.pop_front();
    }

    if (aRotatedArcSegmentSpace->Path()->End().Y() > 0) {
        auto tCircleIt = ++(tEndMeshPoints.fData.begin());
        while (tCircleIt != tEndMeshPoints.fData.end()) {
            tMeshPoints.fData.push_back(*tCircleIt);
            ++tCircleIt;
        }
    }
    else {
        tMeshPoints.fData.pop_back();
    }

    TubeMeshToVTK(tStartApex, tMeshPoints, tEndApex);

    //clear space
    fCurrentSpace = nullptr;

    return;
}

void KGVTKGeometryPainter::VisitRotatedOpenPathSpace(KGRotatedPolyLineSpace* aRotatedPolyLineSpace)
{
    //create line segment points
    OpenPoints tPolyLinePoints;
    PolyLineToOpenPoints(aRotatedPolyLineSpace->Path().operator->(), tPolyLinePoints);

    //create rotated points
    TubeMesh tMeshPoints;
    OpenPointsRotatedToTubeMesh(tPolyLinePoints, tMeshPoints);

    //create start circle points
    ClosedPoints tStartCirclePoints;
    CircleToClosedPoints(aRotatedPolyLineSpace->StartPath().operator->(), tStartCirclePoints);

    //create start flattened mesh points
    TubeMesh tStartMeshPoints;
    KThreeVector tStartApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tStartCirclePoints,
                                           aRotatedPolyLineSpace->StartPath()->Centroid(),
                                           aRotatedPolyLineSpace->Path()->Start().X(),
                                           tStartMeshPoints,
                                           tStartApex);

    //create end circle points
    ClosedPoints tEndCirclePoints;
    CircleToClosedPoints(aRotatedPolyLineSpace->EndPath().operator->(), tEndCirclePoints);

    //create end flattened mesh points
    TubeMesh tEndMeshPoints;
    KThreeVector tEndApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tEndCirclePoints,
                                           aRotatedPolyLineSpace->EndPath()->Centroid(),
                                           aRotatedPolyLineSpace->Path()->End().X(),
                                           tEndMeshPoints,
                                           tEndApex);

    //surgery
    if (aRotatedPolyLineSpace->Path()->Start().Y() > 0) {
        auto tCircleIt = ++(tStartMeshPoints.fData.begin());
        while (tCircleIt != tStartMeshPoints.fData.end()) {
            tMeshPoints.fData.push_front(*tCircleIt);
            ++tCircleIt;
        }
    }
    else {
        tMeshPoints.fData.pop_front();
    }

    if (aRotatedPolyLineSpace->Path()->End().Y() > 0) {
        auto tCircleIt = ++(tEndMeshPoints.fData.begin());
        while (tCircleIt != tEndMeshPoints.fData.end()) {
            tMeshPoints.fData.push_back(*tCircleIt);
            ++tCircleIt;
        }
    }
    else {
        tMeshPoints.fData.pop_back();
    }

    TubeMeshToVTK(tStartApex, tMeshPoints, tEndApex);

    //clear space
    fCurrentSpace = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitRotatedClosedPathSpace(KGRotatedCircleSpace* aRotatedCircleSpace)
{
    //create poly line points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aRotatedCircleSpace->Path().operator->(), tCirclePoints);

    //create rotated points
    TorusMesh tMeshPoints;
    ClosedPointsRotatedToTorusMesh(tCirclePoints, tMeshPoints);

    //create mesh
    TorusMeshToVTK(tMeshPoints);

    return;
}
void KGVTKGeometryPainter::VisitRotatedClosedPathSpace(KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace)
{
    //create poly line points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aRotatedPolyLoopSpace->Path().operator->(), tPolyLoopPoints);

    //create rotated points
    TorusMesh tMeshPoints;
    ClosedPointsRotatedToTorusMesh(tPolyLoopPoints, tMeshPoints);

    //create mesh
    TorusMeshToVTK(tMeshPoints);

    //clear space
    fCurrentSpace = nullptr;

    return;
}
void KGVTKGeometryPainter::VisitExtrudedClosedPathSpace(KGExtrudedCircleSpace* aExtrudedCircleSpace)
{
    //create circle points
    ClosedPoints tCirclePoints;
    CircleToClosedPoints(aExtrudedCircleSpace->Path().operator->(), tCirclePoints);

    //create extruded points
    TubeMesh tMeshPoints;
    ClosedPointsExtrudedToTubeMesh(tCirclePoints,
                                   aExtrudedCircleSpace->ZMin(),
                                   aExtrudedCircleSpace->ZMax(),
                                   tMeshPoints);

    //create start flattened mesh points
    TubeMesh tStartMeshPoints;
    KThreeVector tStartApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tCirclePoints,
                                           aExtrudedCircleSpace->Path()->Centroid(),
                                           aExtrudedCircleSpace->ZMin(),
                                           tStartMeshPoints,
                                           tStartApex);

    //create end flattened mesh points
    TubeMesh tEndMeshPoints;
    KThreeVector tEndApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tCirclePoints,
                                           aExtrudedCircleSpace->Path()->Centroid(),
                                           aExtrudedCircleSpace->ZMax(),
                                           tEndMeshPoints,
                                           tEndApex);

    //surgery
    tMeshPoints.fData.pop_front();
    for (auto& tStartIt : tStartMeshPoints.fData) {
        tMeshPoints.fData.push_front(tStartIt);
    }

    tMeshPoints.fData.pop_back();
    for (auto& tEndIt : tEndMeshPoints.fData) {
        tMeshPoints.fData.push_back(tEndIt);
    }

    //create mesh
    TubeMeshToVTK(tStartApex, tMeshPoints, tEndApex);

    //clear space
    fCurrentSpace = nullptr;

    return;
}

void KGVTKGeometryPainter::VisitExtrudedClosedPathSpace(KGExtrudedPolyLoopSpace* aExtrudedPolyLoopSpace)
{
    //create circle points
    ClosedPoints tPolyLoopPoints;
    PolyLoopToClosedPoints(aExtrudedPolyLoopSpace->Path().operator->(), tPolyLoopPoints);

    //create extruded points
    TubeMesh tMeshPoints;
    ClosedPointsExtrudedToTubeMesh(tPolyLoopPoints,
                                   aExtrudedPolyLoopSpace->ZMin(),
                                   aExtrudedPolyLoopSpace->ZMax(),
                                   tMeshPoints);

    //create start flattened mesh points
    TubeMesh tStartMeshPoints;
    KThreeVector tStartApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tPolyLoopPoints,
                                           aExtrudedPolyLoopSpace->Path()->Centroid(),
                                           aExtrudedPolyLoopSpace->ZMin(),
                                           tStartMeshPoints,
                                           tStartApex);

    //create end flattened mesh points
    TubeMesh tEndMeshPoints;
    KThreeVector tEndApex;
    ClosedPointsFlattenedToTubeMeshAndApex(tPolyLoopPoints,
                                           aExtrudedPolyLoopSpace->Path()->Centroid(),
                                           aExtrudedPolyLoopSpace->ZMax(),
                                           tEndMeshPoints,
                                           tEndApex);

    //surgery
    tMeshPoints.fData.pop_front();
    for (auto& tStartIt : tStartMeshPoints.fData) {
        tMeshPoints.fData.push_front(tStartIt);
    }

    tMeshPoints.fData.pop_back();
    for (auto& tEndIt : tEndMeshPoints.fData) {
        tMeshPoints.fData.push_back(tEndIt);
    }

    //create mesh
    TubeMeshToVTK(tStartApex, tMeshPoints, tEndApex);

    //clear space
    fCurrentSpace = nullptr;

    return;
}

void KGVTKGeometryPainter::VisitWrappedSpace(KGRodSpace* aRodSpace)
{
    if (fIgnore == true) {
        return;
    }

    // getting the radius of the rod elements
    double tRodRadius = aRodSpace->GetObject()->GetRadius();

    // create three-dimensional poly line points
    ThreePoints tRodThreePoints;
    RodsToThreePoints(aRodSpace, tRodThreePoints);

    //create rotated points
    TubeMesh tMeshPoints;
    ThreePointsToTubeMesh(tRodThreePoints, tMeshPoints, tRodRadius);

    TubeMeshToVTK(tMeshPoints);

    //clear space
    fCurrentSpace = nullptr;

    return;
}

void KGVTKGeometryPainter::LocalToGlobal(const KThreeVector& aLocal, KThreeVector& aGlobal)
{
    aGlobal = fCurrentOrigin + aLocal.X() * fCurrentXAxis + aLocal.Y() * fCurrentYAxis + aLocal.Z() * fCurrentZAxis;
    return;
}

//****************
//points functions
//****************

void KGVTKGeometryPainter::LineSegmentToOpenPoints(const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints)
{
    aPoints.fData.clear();

    aPoints.fData.push_back(aLineSegment->At(0.));
    aPoints.fData.push_back(aLineSegment->At(aLineSegment->Length()));

    vismsg_debug("line segment partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom);

    return;
}
void KGVTKGeometryPainter::ArcSegmentToOpenPoints(const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints)
{
    aPoints.fData.clear();

    double tArcFraction = anArcSegment->Length() / (2. * katrin::KConst::Pi() * anArcSegment->Radius());
    auto tArc = (unsigned int) (ceil(tArcFraction * (double) (fCurrentData->GetArc())));

    double tFraction;
    unsigned int tCount;
    for (tCount = 0; tCount <= tArc; tCount++) {
        tFraction = anArcSegment->Length() * ((double) (tCount) / (double) (tArc));
        aPoints.fData.push_back(anArcSegment->At(tFraction));
    }

    vismsg_debug("arc segment partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom);

    return;
}
void KGVTKGeometryPainter::PolyLineToOpenPoints(const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints)
{
    aPoints.fData.clear();

    const KGPlanarPolyLine::Set& tElements = aPolyLine->Elements();
    KGPlanarPolyLine::CIt tElementIt;
    const KGPlanarOpenPath* tElement;
    const KGPlanarLineSegment* tLineSegmentElement;
    const KGPlanarArcSegment* tArcSegmentElement;

    OpenPoints tSubPoints;
    for (tElementIt = tElements.begin(); tElementIt != tElements.end(); tElementIt++) {
        tElement = *tElementIt;

        tLineSegmentElement = dynamic_cast<const KGPlanarLineSegment*>(tElement);
        if (tLineSegmentElement != nullptr) {
            LineSegmentToOpenPoints(tLineSegmentElement, tSubPoints);
            aPoints.fData.insert(aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()));
            continue;
        }

        tArcSegmentElement = dynamic_cast<const KGPlanarArcSegment*>(tElement);
        if (tArcSegmentElement != nullptr) {
            ArcSegmentToOpenPoints(tArcSegmentElement, tSubPoints);
            aPoints.fData.insert(aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()));
            continue;
        }
    }

    aPoints.fData.push_back(aPolyLine->End());

    vismsg_debug("poly line partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom);

    return;
}
void KGVTKGeometryPainter::CircleToClosedPoints(const KGPlanarCircle* aCircle, ClosedPoints& aPoints)
{
    aPoints.fData.clear();

    unsigned int tArc = fCurrentData->GetArc();

    double tFraction;
    unsigned int tCount;
    for (tCount = 0; tCount < tArc; tCount++) {
        tFraction = aCircle->Length() * ((double) (tCount) / (double) (tArc));
        aPoints.fData.push_back(aCircle->At(tFraction));
    }

    vismsg_debug("circle partitioned into <" << aPoints.fData.size() << "> closed points vertices" << eom);

    return;
}
void KGVTKGeometryPainter::PolyLoopToClosedPoints(const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints)
{
    aPoints.fData.clear();

    const KGPlanarPolyLoop::Set& tElements = aPolyLoop->Elements();
    KGPlanarPolyLoop::CIt tElementIt;
    const KGPlanarOpenPath* tElement;
    const KGPlanarLineSegment* tLineSegmentElement;
    const KGPlanarArcSegment* tArcSegmentElement;

    OpenPoints tSubPoints;
    for (tElementIt = tElements.begin(); tElementIt != tElements.end(); tElementIt++) {
        tElement = *tElementIt;
        tSubPoints.fData.clear();

        tLineSegmentElement = dynamic_cast<const KGPlanarLineSegment*>(tElement);
        if (tLineSegmentElement != nullptr) {
            LineSegmentToOpenPoints(tLineSegmentElement, tSubPoints);
            aPoints.fData.insert(aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()));
            continue;
        }

        tArcSegmentElement = dynamic_cast<const KGPlanarArcSegment*>(tElement);
        if (tArcSegmentElement != nullptr) {
            ArcSegmentToOpenPoints(tArcSegmentElement, tSubPoints);
            aPoints.fData.insert(aPoints.fData.end(), tSubPoints.fData.begin(), --(tSubPoints.fData.end()));
            continue;
        }
    }

    vismsg_debug("poly loop partitioned into <" << aPoints.fData.size() << "> open points vertices" << eom);

    return;
}

void KGVTKGeometryPainter::RodsToThreePoints(const KGRodSpace* aRodSpace, ThreePoints& aThreePoints)
{
    aThreePoints.fData.clear();

    // determine number of rod vertices
    unsigned int tNCoordinates = aRodSpace->GetObject()->GetNCoordinates();

    // 1 SubThreePoint object = 2 Vertices
    ThreePoints tSubThreePoints;
    KThreeVector tStartPoint;
    KThreeVector tEndPoint;

    for (unsigned int tCoordinateIt = 0; tCoordinateIt < (tNCoordinates - 1); tCoordinateIt++) {
        tSubThreePoints.fData.clear();

        tStartPoint.SetComponents(aRodSpace->GetObject()->GetCoordinate(tCoordinateIt)[0],
                                  aRodSpace->GetObject()->GetCoordinate(tCoordinateIt)[1],
                                  aRodSpace->GetObject()->GetCoordinate(tCoordinateIt)[2]);
        tEndPoint.SetComponents(aRodSpace->GetObject()->GetCoordinate(tCoordinateIt + 1, 0),
                                aRodSpace->GetObject()->GetCoordinate(tCoordinateIt + 1, 1),
                                aRodSpace->GetObject()->GetCoordinate(tCoordinateIt + 1, 2));

        tSubThreePoints.fData.push_back(tStartPoint);
        tSubThreePoints.fData.push_back(tEndPoint);

        aThreePoints.fData.insert(aThreePoints.fData.end(), tSubThreePoints.fData.begin(), tSubThreePoints.fData.end());
    }

    return;
}

void KGVTKGeometryPainter::RodsToThreePoints(const KGRodSurface* aRodSurface, ThreePoints& aThreePoints)
{
    aThreePoints.fData.clear();

    // determine number of rod vertices
    unsigned int tNCoordinates = aRodSurface->GetObject()->GetNCoordinates();

    // 1 SubThreePoint object = 2 Vertices
    ThreePoints tSubThreePoints;
    KThreeVector tStartPoint;
    KThreeVector tEndPoint;

    for (unsigned int tCoordinateIt = 0; tCoordinateIt < (tNCoordinates - 1); tCoordinateIt++) {
        tSubThreePoints.fData.clear();

        tStartPoint.SetComponents(aRodSurface->GetObject()->GetCoordinate(tCoordinateIt)[0],
                                  aRodSurface->GetObject()->GetCoordinate(tCoordinateIt)[1],
                                  aRodSurface->GetObject()->GetCoordinate(tCoordinateIt)[2]);
        tEndPoint.SetComponents(aRodSurface->GetObject()->GetCoordinate(tCoordinateIt + 1, 0),
                                aRodSurface->GetObject()->GetCoordinate(tCoordinateIt + 1, 1),
                                aRodSurface->GetObject()->GetCoordinate(tCoordinateIt + 1, 2));

        tSubThreePoints.fData.push_back(tStartPoint);
        tSubThreePoints.fData.push_back(tEndPoint);

        aThreePoints.fData.insert(aThreePoints.fData.end(), tSubThreePoints.fData.begin(), tSubThreePoints.fData.end());
    }

    return;
}

void KGVTKGeometryPainter::WireArrayToThreePoints(const KGConicalWireArraySurface* aConicalWireArraySurface,
                                                  ThreePoints& aThreePoints)
{
    aThreePoints.fData.clear();

    // gathering data from wire array object
    unsigned int tWiresInArray = aConicalWireArraySurface->GetObject()->GetNWires();
    double tThetaOffset = aConicalWireArraySurface->GetObject()->GetThetaStart() * (katrin::KConst::Pi() / 180.);

    double tZ1 = aConicalWireArraySurface->GetObject()->GetZ1();
    double tR1 = aConicalWireArraySurface->GetObject()->GetR1();
    double tZ2 = aConicalWireArraySurface->GetObject()->GetZ2();
    double tR2 = aConicalWireArraySurface->GetObject()->GetR2();

    double tAngleStep = 2 * katrin::KConst::Pi() / tWiresInArray;

    // 1 SubThreePoint object = 2 Vertices
    ThreePoints tSubThreePoints;
    KThreeVector tStartPoint;
    KThreeVector tEndPoint;

    for (unsigned int tWireIt = 0; tWireIt < tWiresInArray; tWireIt++) {
        tSubThreePoints.fData.clear();

        tStartPoint.SetComponents(tR1 * cos(tWireIt * tAngleStep + tThetaOffset),
                                  tR1 * sin(tWireIt * tAngleStep + tThetaOffset),
                                  tZ1);
        tEndPoint.SetComponents(tR2 * cos(tWireIt * tAngleStep + tThetaOffset),
                                tR2 * sin(tWireIt * tAngleStep + tThetaOffset),
                                tZ2);

        tSubThreePoints.fData.push_back(tStartPoint);
        tSubThreePoints.fData.push_back(tEndPoint);

        aThreePoints.fData.insert(aThreePoints.fData.end(), tSubThreePoints.fData.begin(), tSubThreePoints.fData.end());
    }

    return;
}

//**************
//mesh functions
//**************

void KGVTKGeometryPainter::ClosedPointsFlattenedToTubeMeshAndApex(const ClosedPoints& aPoints,
                                                                  const KTwoVector& aCentroid, const double& aZ,
                                                                  TubeMesh& aMesh, KThreeVector& anApex)
{
    KThreeVector tPoint;
    TubeMesh::Group tGroup;
    for (const auto& tPointsIt : aPoints.fData) {
        tPoint.X() = tPointsIt.X();
        tPoint.Y() = tPointsIt.Y();
        tPoint.Z() = aZ;
        tGroup.push_back(tPoint);
    }
    aMesh.fData.push_back(tGroup);
    anApex.X() = aCentroid.X();
    anApex.Y() = aCentroid.Y();
    anApex.Z() = aZ;

    vismsg_debug("flattened closed points into <" << aMesh.fData.size() * aMesh.fData.front().size()
                                                  << "> tube mesh vertices" << eom);

    return;
}
void KGVTKGeometryPainter::OpenPointsRotatedToTubeMesh(const OpenPoints& aPoints, TubeMesh& aMesh)
{
    unsigned int tArc = fCurrentData->GetArc();

    double tFraction;
    unsigned int tCount;

    KThreeVector tPoint;
    TubeMesh::Group tGroup;
    for (const auto& tPointsIt : aPoints.fData) {
        tGroup.clear();
        for (tCount = 0; tCount < tArc; tCount++) {
            tFraction = (double) (tCount) / (double) (tArc);

            tPoint.X() = tPointsIt.Y() * cos(2. * katrin::KConst::Pi() * tFraction);
            tPoint.Y() = tPointsIt.Y() * sin(2. * katrin::KConst::Pi() * tFraction);
            tPoint.Z() = tPointsIt.X();
            tGroup.push_back(tPoint);
        }
        aMesh.fData.push_back(tGroup);
    }

    vismsg_debug("rotated open points into <" << aMesh.fData.size() * aMesh.fData.front().size()
                                              << "> tube mesh vertices" << eom);

    return;
}
void KGVTKGeometryPainter::OpenPointsRotatedToShellMesh(const OpenPoints& aPoints, ShellMesh& aMesh,
                                                        const double& aAngleStart, const double& aAngleStop)
{
    unsigned int tArc = fCurrentData->GetArc();

    double tFraction;
    unsigned int tCount;
    double tAngle = (aAngleStop - aAngleStart) / 360;

    KThreeVector tPoint;
    ShellMesh::Group tGroup;
    for (const auto& tPointsIt : aPoints.fData) {
        tGroup.clear();
        for (tCount = 0; tCount <= tArc; tCount++) {
            tFraction = (double) (tCount) / (double) (tArc);

            tPoint.X() = tPointsIt.Y() * cos(2. * katrin::KConst::Pi() * tFraction * tAngle +
                                             aAngleStart * katrin::KConst::Pi() / 180.);
            tPoint.Y() = tPointsIt.Y() * sin(2. * katrin::KConst::Pi() * tFraction * tAngle +
                                             aAngleStart * katrin::KConst::Pi() / 180.);
            tPoint.Z() = tPointsIt.X();
            tGroup.push_back(tPoint);
        }
        aMesh.fData.push_back(tGroup);
    }


    vismsg_debug("rotated open points into <" << aMesh.fData.size() * aMesh.fData.front().size()
                                              << "> tube mesh vertices" << eom);

    return;
}

void KGVTKGeometryPainter::ClosedPointsRotatedToTorusMesh(const ClosedPoints& aPoints, TorusMesh& aMesh)
{
    unsigned int tArc = fCurrentData->GetArc();

    double tFraction;
    unsigned int tCount;

    KThreeVector tPoint;
    TubeMesh::Group tGroup;
    for (const auto& tPointsIt : aPoints.fData) {
        tGroup.clear();
        for (tCount = 0; tCount < tArc; tCount++) {
            tFraction = (double) (tCount) / (double) (tArc);

            tPoint.X() = tPointsIt.Y() * cos(2. * katrin::KConst::Pi() * tFraction);
            tPoint.Y() = tPointsIt.Y() * sin(2. * katrin::KConst::Pi() * tFraction);
            tPoint.Z() = tPointsIt.X();
            tGroup.push_back(tPoint);
        }
        aMesh.fData.push_back(tGroup);
    }

    vismsg_debug("rotated closed points into <" << aMesh.fData.size() * aMesh.fData.front().size()
                                                << "> torus mesh vertices" << eom);

    return;
}
void KGVTKGeometryPainter::ClosedPointsRotatedToShellMesh(const ClosedPoints& aPoints, ShellMesh& aMesh,
                                                          const double& aAngleStart, const double& aAngleStop)
{
    unsigned int tArc = fCurrentData->GetArc();

    double tFraction;
    unsigned int tCount;
    double tAngle = (aAngleStop - aAngleStart) / 360;
    KThreeVector tPoint;
    TubeMesh::Group tGroup;
    for (const auto& tPointsIt : aPoints.fData) {
        tGroup.clear();
        for (tCount = 0; tCount <= tArc; tCount++) {
            tFraction = (double) (tCount) / (double) (tArc);

            tPoint.X() = tPointsIt.Y() * cos(2. * katrin::KConst::Pi() * tFraction * tAngle +
                                             aAngleStart * katrin::KConst::Pi() / 180.);
            tPoint.Y() = tPointsIt.Y() * sin(2. * katrin::KConst::Pi() * tFraction * tAngle +
                                             aAngleStart * katrin::KConst::Pi() / 180.);
            tPoint.Z() = tPointsIt.X();
            tGroup.push_back(tPoint);
        }
        aMesh.fData.push_back(tGroup);
    }

    vismsg_debug("rotated closed points into <" << aMesh.fData.size() * aMesh.fData.front().size()
                                                << "> torus mesh vertices" << eom);

    return;
}
void KGVTKGeometryPainter::OpenPointsExtrudedToFlatMesh(const OpenPoints& aPoints, const double& aZMin,
                                                        const double& aZMax, FlatMesh& aMesh)
{
    KThreeVector tPoint;
    TubeMesh::Group tGroup;

    tGroup.clear();
    for (const auto& tPointsIt : aPoints.fData) {
        tPoint.X() = tPointsIt.X();
        tPoint.Y() = tPointsIt.Y();
        tPoint.Z() = aZMin;
        tGroup.push_back(tPoint);
    }
    aMesh.fData.push_back(tGroup);

    tGroup.clear();
    for (const auto& tPointsIt : aPoints.fData) {
        tPoint.X() = tPointsIt.X();
        tPoint.Y() = tPointsIt.Y();
        tPoint.Z() = aZMax;
        tGroup.push_back(tPoint);
    }
    aMesh.fData.push_back(tGroup);

    vismsg_debug("extruded open points into <" << aMesh.fData.size() * aMesh.fData.front().size()
                                               << "> flat mesh vertices" << eom);

    return;
}
void KGVTKGeometryPainter::ClosedPointsExtrudedToTubeMesh(const ClosedPoints& aPoints, const double& aZMin,
                                                          const double& aZMax, TubeMesh& aMesh)
{
    KThreeVector tPoint;
    TubeMesh::Group tGroup;

    tGroup.clear();
    for (const auto& tPointsIt : aPoints.fData) {
        tPoint.X() = tPointsIt.X();
        tPoint.Y() = tPointsIt.Y();
        tPoint.Z() = aZMin;
        tGroup.push_back(tPoint);
    }
    aMesh.fData.push_back(tGroup);

    tGroup.clear();
    for (const auto& tPointsIt : aPoints.fData) {
        tPoint.X() = tPointsIt.X();
        tPoint.Y() = tPointsIt.Y();
        tPoint.Z() = aZMax;
        tGroup.push_back(tPoint);
    }
    aMesh.fData.push_back(tGroup);

    vismsg_debug("extruded closed points into <" << aMesh.fData.size() * aMesh.fData.front().size()
                                                 << "> tube mesh vertices" << eom);

    return;
}
void KGVTKGeometryPainter::ThreePointsToTubeMesh(const ThreePoints& aThreePoints, TubeMesh& aMesh,
                                                 const double& aTubeRadius)
{
    unsigned int tArc = fCurrentData->GetArc();

    double tFraction;
    unsigned int tCount;

    KThreeVector tEndPoint;
    KThreeVector tStartPoint;
    KThreeVector tCenterPoint;
    KThreeVector tAxisUnit;

    double toDegree = 180. / katrin::KConst::Pi();
    double tTheta(0.);
    double tPhi(0.);
    double tR(0.);

    KTransformation tEulerZXZ;

    KThreeVector tPoint;
    TubeMesh::Group tGroup;

    for (auto tThreePointsIt = aThreePoints.fData.begin(); tThreePointsIt != aThreePoints.fData.end();
         tThreePointsIt++) {
        // define start and end point
        tStartPoint = *tThreePointsIt;
        tThreePointsIt++;
        tEndPoint = *tThreePointsIt;

        // determine center of rod axis
        tCenterPoint = (tEndPoint + tStartPoint) / 2.;

        // rod axis unit vector
        tAxisUnit = (tEndPoint - tStartPoint).Unit();
        if (fabs(tAxisUnit.GetZ()) > 1. && tAxisUnit.GetZ() > 0.)
            tAxisUnit.SetZ(1.);
        if (fabs(tAxisUnit.GetZ()) > 1. && tAxisUnit.GetZ() < 0.)
            tAxisUnit.SetZ(-1.);

        // computation of Euler angles
        tR = sqrt(tAxisUnit.GetX() * tAxisUnit.GetX() + tAxisUnit.GetY() * tAxisUnit.GetY());
        tTheta = toDegree * acos(tAxisUnit.GetZ());

        if (tR < 1.e-10)
            tPhi = 0.;
        else {
            if (tAxisUnit.GetX() >= 0. && tAxisUnit.GetY() >= 0.)
                tPhi = 180. - toDegree * asin(tAxisUnit.GetX() / tR);
            else if (tAxisUnit.GetX() >= 0. && tAxisUnit.GetY() <= 0.)
                tPhi = toDegree * asin(tAxisUnit.GetX() / tR);
            else if (tAxisUnit.GetX() <= 0. && tAxisUnit.GetY() >= 0.)
                tPhi = 180. + toDegree * asin(-tAxisUnit.GetX() / tR);
            else
                tPhi = 360. - toDegree * asin(-tAxisUnit.GetX() / tR);
        }

        tEulerZXZ.SetRotationEuler(tPhi, tTheta, 0.);


        // computing and setting mesh points
        tGroup.clear();
        for (tCount = 0; tCount < tArc; tCount++) {
            tEulerZXZ.SetOrigin(tStartPoint);
            tFraction = (double) (tCount) / (double) (tArc);

            tPoint.X() = (tStartPoint.X() + aTubeRadius * cos(2. * katrin::KConst::Pi() * tFraction));
            tPoint.Y() = (tStartPoint.Y() + aTubeRadius * sin(2. * katrin::KConst::Pi() * tFraction));
            tPoint.Z() = (tStartPoint.Z());

            tEulerZXZ.ApplyRotation(tPoint);

            tGroup.push_back(tPoint);
        }
        aMesh.fData.push_back(tGroup);

        tGroup.clear();
        for (tCount = 0; tCount < tArc; tCount++) {
            tEulerZXZ.SetOrigin(tEndPoint);
            tFraction = (double) (tCount) / (double) (tArc);

            tPoint.X() = (tEndPoint.X() + aTubeRadius * cos(2. * katrin::KConst::Pi() * tFraction));
            tPoint.Y() = (tEndPoint.Y() + aTubeRadius * sin(2. * katrin::KConst::Pi() * tFraction));
            tPoint.Z() = (tEndPoint.Z());

            tEulerZXZ.ApplyRotation(tPoint);

            tGroup.push_back(tPoint);
        }
        aMesh.fData.push_back(tGroup);
    }
    vismsg_debug("rotated open points into <" << aMesh.fData.size() * aMesh.fData.front().size()
                                              << "> tube mesh vertices" << eom);

    return;
}
void KGVTKGeometryPainter::ThreePointsToTubeMeshToVTK(const ThreePoints& aThreePoints, TubeMesh& aMesh,
                                                      const double& aTubeRadius)
{
    unsigned int tArc = fCurrentData->GetArc();

    double tFraction;
    unsigned int tCount;

    KThreeVector tEndPoint;
    KThreeVector tStartPoint;
    KThreeVector tCenterPoint;
    KThreeVector tAxisUnit;

    double toDegree = 180. / katrin::KConst::Pi();
    double tTheta(0.);
    double tPhi(0.);
    double tR(0.);

    KTransformation tEulerZXZ;

    KThreeVector tPoint;
    TubeMesh::Group tGroup;

    for (auto tThreePointsIt = aThreePoints.fData.begin(); tThreePointsIt != aThreePoints.fData.end();
         tThreePointsIt++) {
        // define start and end point
        tStartPoint = *tThreePointsIt;
        tThreePointsIt++;
        tEndPoint = *tThreePointsIt;

        // determine center of rod axis
        tCenterPoint = (tEndPoint + tStartPoint) / 2.;

        // rod axis unit vector
        tAxisUnit = (tEndPoint - tStartPoint).Unit();
        if (fabs(tAxisUnit.GetZ()) > 1. && tAxisUnit.GetZ() > 0.)
            tAxisUnit.SetZ(1.);
        if (fabs(tAxisUnit.GetZ()) > 1. && tAxisUnit.GetZ() < 0.)
            tAxisUnit.SetZ(-1.);

        // computation of Euler angles
        tR = sqrt(tAxisUnit.GetX() * tAxisUnit.GetX() + tAxisUnit.GetY() * tAxisUnit.GetY());
        tTheta = toDegree * acos(tAxisUnit.GetZ());

        if (tR < 1.e-10)
            tPhi = 0.;
        else {
            if (tAxisUnit.GetX() >= 0. && tAxisUnit.GetY() >= 0.)
                tPhi = 180. - toDegree * asin(tAxisUnit.GetX() / tR);
            else if (tAxisUnit.GetX() >= 0. && tAxisUnit.GetY() <= 0.)
                tPhi = toDegree * asin(tAxisUnit.GetX() / tR);
            else if (tAxisUnit.GetX() <= 0. && tAxisUnit.GetY() >= 0.)
                tPhi = 180. + toDegree * asin(-tAxisUnit.GetX() / tR);
            else
                tPhi = 360. - toDegree * asin(-tAxisUnit.GetX() / tR);
        }

        tEulerZXZ.SetRotationEuler(tPhi, tTheta, 0.);

        // computing and setting mesh points
        tGroup.clear();
        for (tCount = 0; tCount < tArc; tCount++) {
            tEulerZXZ.SetOrigin(tStartPoint);
            tFraction = (double) (tCount) / (double) (tArc);

            tPoint.X() = (tStartPoint.X() + aTubeRadius * cos(2. * katrin::KConst::Pi() * tFraction));
            tPoint.Y() = (tStartPoint.Y() + aTubeRadius * sin(2. * katrin::KConst::Pi() * tFraction));
            tPoint.Z() = (tStartPoint.Z());

            tEulerZXZ.ApplyRotation(tPoint);

            tGroup.push_back(tPoint);
        }
        aMesh.fData.push_back(tGroup);

        tGroup.clear();
        for (tCount = 0; tCount < tArc; tCount++) {
            tEulerZXZ.SetOrigin(tEndPoint);
            tFraction = (double) (tCount) / (double) (tArc);

            tPoint.X() = (tEndPoint.X() + aTubeRadius * cos(2. * katrin::KConst::Pi() * tFraction));
            tPoint.Y() = (tEndPoint.Y() + aTubeRadius * sin(2. * katrin::KConst::Pi() * tFraction));
            tPoint.Z() = (tEndPoint.Z());

            tEulerZXZ.ApplyRotation(tPoint);

            tGroup.push_back(tPoint);
        }
        aMesh.fData.push_back(tGroup);

        // rendering
        TubeMeshToVTK(aMesh);

        aMesh.fData.clear();
    }
    vismsg_debug("rotated open points into <" << aMesh.fData.size() * aMesh.fData.front().size()
                                              << "> tube mesh vertices" << eom);

    return;
}

//*******************
//rendering functions
//*******************

void KGVTKGeometryPainter::MeshToVTK(const Mesh& aMesh)
{
    //object allocation
    KThreeVector tPoint;

    deque<vtkIdType> vMeshIdGroup;
    deque<deque<vtkIdType>> vMeshIdSet;

    deque<vtkIdType>::iterator vThisPoint;
    deque<deque<vtkIdType>>::iterator vThisGroup;

    vtkSmartPointer<vtkCell> vCell;

    //create mesh point ids
    for (const auto& tSetIt : aMesh.fData) {
        vMeshIdGroup.clear();
        for (const auto& tGroupIt : tSetIt) {
            LocalToGlobal(tGroupIt, tPoint);
            vMeshIdGroup.push_back(fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z()));
        }
        vMeshIdSet.push_back(vMeshIdGroup);
    }

    //create hull cells
    vThisGroup = vMeshIdSet.begin();

    while (vThisGroup != vMeshIdSet.end()) {
        vThisPoint = vThisGroup->begin();
        if (vThisGroup->size() <= 2)
            continue;
        else if (vThisGroup->size() == 3)
            vCell = vtkSmartPointer<vtkTriangle>::New();
        else if (vThisGroup->size() == 4)
            vCell = vtkSmartPointer<vtkQuad>::New();
        else
            vCell = vtkSmartPointer<vtkPolygon>::New();

        for (unsigned i = 0; i < vThisGroup->size(); i++)
            vCell->GetPointIds()->SetId(i, *(vThisPoint++));
        fCells->InsertNextCell(vCell);
        fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                  fCurrentData->GetColor().GetGreen(),
                                  fCurrentData->GetColor().GetBlue(),
                                  fCurrentData->GetColor().GetOpacity());

        ++vThisGroup;
    }

    return;
}

void KGVTKGeometryPainter::FlatMeshToVTK(const FlatMesh& aMesh)
{
    //object allocation
    KThreeVector tPoint;

    deque<vtkIdType> vMeshIdGroup;
    deque<deque<vtkIdType>> vMeshIdSet;

    deque<deque<vtkIdType>>::iterator vThisGroup;
    deque<deque<vtkIdType>>::iterator vNextGroup;

    deque<vtkIdType>::iterator vThisThisPoint;
    deque<vtkIdType>::iterator vThisNextPoint;
    deque<vtkIdType>::iterator vNextThisPoint;
    deque<vtkIdType>::iterator vNextNextPoint;

    vtkSmartPointer<vtkQuad> vQuad;

    //create mesh point ids
    for (const auto& tSetIt : aMesh.fData) {
        vMeshIdGroup.clear();
        for (const auto& tGroupIt : tSetIt) {
            LocalToGlobal(tGroupIt, tPoint);
            vMeshIdGroup.push_back(fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z()));
        }
        vMeshIdSet.push_back(vMeshIdGroup);
    }

    //create hull cells
    vThisGroup = vMeshIdSet.begin();
    vNextGroup = ++(vMeshIdSet.begin());
    while (vNextGroup != vMeshIdSet.end()) {
        vThisThisPoint = vThisGroup->begin();
        vThisNextPoint = ++(vThisGroup->begin());
        vNextThisPoint = vNextGroup->begin();
        vNextNextPoint = ++(vNextGroup->begin());

        while (vNextNextPoint != vNextGroup->end()) {
            vQuad = vtkSmartPointer<vtkQuad>::New();
            vQuad->GetPointIds()->SetId(0, *vThisThisPoint);
            vQuad->GetPointIds()->SetId(1, *vNextThisPoint);
            vQuad->GetPointIds()->SetId(2, *vNextNextPoint);
            vQuad->GetPointIds()->SetId(3, *vThisNextPoint);
            fCells->InsertNextCell(vQuad);
            fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                      fCurrentData->GetColor().GetGreen(),
                                      fCurrentData->GetColor().GetBlue(),
                                      fCurrentData->GetColor().GetOpacity());
            ++vThisThisPoint;
            ++vThisNextPoint;
            ++vNextThisPoint;
            ++vNextNextPoint;
        }

        ++vThisGroup;
        ++vNextGroup;
    }

    return;
}
void KGVTKGeometryPainter::TubeMeshToVTK(const TubeMesh& aMesh)
{
    //object allocation
    KThreeVector tPoint;

    deque<vtkIdType> vMeshIdGroup;
    deque<deque<vtkIdType>> vMeshIdSet;

    deque<deque<vtkIdType>>::iterator vThisGroup;
    deque<deque<vtkIdType>>::iterator vNextGroup;

    deque<vtkIdType>::iterator vThisThisPoint;
    deque<vtkIdType>::iterator vThisNextPoint;
    deque<vtkIdType>::iterator vNextThisPoint;
    deque<vtkIdType>::iterator vNextNextPoint;

    vtkSmartPointer<vtkQuad> vQuad;

    //create mesh point ids
    for (const auto& tSetIt : aMesh.fData) {
        vMeshIdGroup.clear();
        for (const auto& tGroupIt : tSetIt) {
            LocalToGlobal(tGroupIt, tPoint);
            vMeshIdGroup.push_back(fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z()));
        }
        vMeshIdGroup.push_back(vMeshIdGroup.front());
        vMeshIdSet.push_back(vMeshIdGroup);
    }

    //create hull cells
    vThisGroup = vMeshIdSet.begin();
    vNextGroup = ++(vMeshIdSet.begin());
    while (vNextGroup != vMeshIdSet.end()) {
        vThisThisPoint = vThisGroup->begin();
        vThisNextPoint = ++(vThisGroup->begin());
        vNextThisPoint = vNextGroup->begin();
        vNextNextPoint = ++(vNextGroup->begin());

        while (vNextNextPoint != vNextGroup->end()) {
            vQuad = vtkSmartPointer<vtkQuad>::New();
            vQuad->GetPointIds()->SetId(0, *vThisThisPoint);
            vQuad->GetPointIds()->SetId(1, *vNextThisPoint);
            vQuad->GetPointIds()->SetId(2, *vNextNextPoint);
            vQuad->GetPointIds()->SetId(3, *vThisNextPoint);
            fCells->InsertNextCell(vQuad);
            fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                      fCurrentData->GetColor().GetGreen(),
                                      fCurrentData->GetColor().GetBlue(),
                                      fCurrentData->GetColor().GetOpacity());
            ++vThisThisPoint;
            ++vThisNextPoint;
            ++vNextThisPoint;
            ++vNextNextPoint;
        }

        ++vThisGroup;
        ++vNextGroup;
    }

    return;
}
void KGVTKGeometryPainter::TubeMeshToVTK(const KThreeVector& anApexStart, const TubeMesh& aMesh)
{
    //object allocation
    KThreeVector tPoint;

    vtkIdType vMeshIdApexStart;
    deque<vtkIdType> vMeshIdGroup;
    deque<deque<vtkIdType>> vMeshIdSet;

    deque<deque<vtkIdType>>::iterator vThisGroup;
    deque<deque<vtkIdType>>::iterator vNextGroup;

    deque<vtkIdType>::iterator vThisThisPoint;
    deque<vtkIdType>::iterator vThisNextPoint;
    deque<vtkIdType>::iterator vNextThisPoint;
    deque<vtkIdType>::iterator vNextNextPoint;

    vtkSmartPointer<vtkTriangle> vTriangle;
    vtkSmartPointer<vtkQuad> vQuad;

    //create apex start point id
    LocalToGlobal(anApexStart, tPoint);
    vMeshIdApexStart = fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z());

    //create mesh point ids
    for (const auto& tSetIt : aMesh.fData) {
        vMeshIdGroup.clear();
        for (const auto& tGroupIt : tSetIt) {
            LocalToGlobal(tGroupIt, tPoint);
            vMeshIdGroup.push_back(fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z()));
        }
        vMeshIdGroup.push_back(vMeshIdGroup.front());
        vMeshIdSet.push_back(vMeshIdGroup);
    }

    //create start cap cells
    vThisThisPoint = vMeshIdSet.front().begin();
    vThisNextPoint = ++(vMeshIdSet.front().begin());
    while (vThisNextPoint != vMeshIdSet.front().end()) {
        vTriangle = vtkSmartPointer<vtkTriangle>::New();
        vTriangle->GetPointIds()->SetId(0, *vThisThisPoint);
        vTriangle->GetPointIds()->SetId(1, *vThisNextPoint);
        vTriangle->GetPointIds()->SetId(2, vMeshIdApexStart);
        fCells->InsertNextCell(vTriangle);
        fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                  fCurrentData->GetColor().GetGreen(),
                                  fCurrentData->GetColor().GetBlue(),
                                  fCurrentData->GetColor().GetOpacity());
        ++vThisThisPoint;
        ++vThisNextPoint;
    }

    //create hull cells
    vThisGroup = vMeshIdSet.begin();
    vNextGroup = ++(vMeshIdSet.begin());
    while (vNextGroup != vMeshIdSet.end()) {
        vThisThisPoint = vThisGroup->begin();
        vThisNextPoint = ++(vThisGroup->begin());
        vNextThisPoint = vNextGroup->begin();
        vNextNextPoint = ++(vNextGroup->begin());

        while (vNextNextPoint != vNextGroup->end()) {
            vQuad = vtkSmartPointer<vtkQuad>::New();
            vQuad->GetPointIds()->SetId(0, *vThisThisPoint);
            vQuad->GetPointIds()->SetId(1, *vNextThisPoint);
            vQuad->GetPointIds()->SetId(2, *vNextNextPoint);
            vQuad->GetPointIds()->SetId(3, *vThisNextPoint);
            fCells->InsertNextCell(vQuad);
            fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                      fCurrentData->GetColor().GetGreen(),
                                      fCurrentData->GetColor().GetBlue(),
                                      fCurrentData->GetColor().GetOpacity());
            ++vThisThisPoint;
            ++vThisNextPoint;
            ++vNextThisPoint;
            ++vNextNextPoint;
        }

        ++vThisGroup;
        ++vNextGroup;
    }

    return;
}
void KGVTKGeometryPainter::TubeMeshToVTK(const TubeMesh& aMesh, const KThreeVector& anApexEnd)
{
    //object allocation
    KThreeVector tPoint;

    vtkIdType vMeshIdApexEnd;
    deque<vtkIdType> vMeshIdGroup;
    deque<deque<vtkIdType>> vMeshIdSet;

    deque<deque<vtkIdType>>::iterator vThisGroup;
    deque<deque<vtkIdType>>::iterator vNextGroup;

    deque<vtkIdType>::iterator vThisThisPoint;
    deque<vtkIdType>::iterator vThisNextPoint;
    deque<vtkIdType>::iterator vNextThisPoint;
    deque<vtkIdType>::iterator vNextNextPoint;

    vtkSmartPointer<vtkTriangle> vTriangle;
    vtkSmartPointer<vtkQuad> vQuad;

    //create apex end point id
    LocalToGlobal(anApexEnd, tPoint);
    vMeshIdApexEnd = fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z());

    //create mesh point ids
    for (const auto& tSetIt : aMesh.fData) {
        vMeshIdGroup.clear();
        for (const auto& tGroupIt : tSetIt) {
            LocalToGlobal(tGroupIt, tPoint);
            vMeshIdGroup.push_back(fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z()));
        }
        vMeshIdGroup.push_back(vMeshIdGroup.front());
        vMeshIdSet.push_back(vMeshIdGroup);
    }

    //create hull cells
    vThisGroup = vMeshIdSet.begin();
    vNextGroup = ++(vMeshIdSet.begin());
    while (vNextGroup != vMeshIdSet.end()) {
        vThisThisPoint = vThisGroup->begin();
        vThisNextPoint = ++(vThisGroup->begin());
        vNextThisPoint = vNextGroup->begin();
        vNextNextPoint = ++(vNextGroup->begin());

        while (vNextNextPoint != vNextGroup->end()) {
            vQuad = vtkSmartPointer<vtkQuad>::New();
            vQuad->GetPointIds()->SetId(0, *vThisThisPoint);
            vQuad->GetPointIds()->SetId(1, *vNextThisPoint);
            vQuad->GetPointIds()->SetId(2, *vNextNextPoint);
            vQuad->GetPointIds()->SetId(3, *vThisNextPoint);
            fCells->InsertNextCell(vQuad);
            fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                      fCurrentData->GetColor().GetGreen(),
                                      fCurrentData->GetColor().GetBlue(),
                                      fCurrentData->GetColor().GetOpacity());
            ++vThisThisPoint;
            ++vThisNextPoint;
            ++vNextThisPoint;
            ++vNextNextPoint;
        }

        ++vThisGroup;
        ++vNextGroup;
    }

    //create end cap cells
    vThisThisPoint = vMeshIdSet.back().begin();
    vThisNextPoint = ++(vMeshIdSet.back().begin());
    while (vThisNextPoint != vMeshIdSet.back().end()) {
        vTriangle = vtkSmartPointer<vtkTriangle>::New();
        vTriangle->GetPointIds()->SetId(0, *vThisThisPoint);
        vTriangle->GetPointIds()->SetId(1, *vThisNextPoint);
        vTriangle->GetPointIds()->SetId(2, vMeshIdApexEnd);
        fCells->InsertNextCell(vTriangle);
        fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                  fCurrentData->GetColor().GetGreen(),
                                  fCurrentData->GetColor().GetBlue(),
                                  fCurrentData->GetColor().GetOpacity());
        ++vThisThisPoint;
        ++vThisNextPoint;
    }

    return;
}
void KGVTKGeometryPainter::TubeMeshToVTK(const KThreeVector& anApexStart, const TubeMesh& aMesh,
                                         const KThreeVector& anApexEnd)
{
    if (aMesh.fData.empty()) {
        vismsg(eWarning) << "mesh has size of zero, check your geometry" << eom;
    }

    //object allocation
    KThreeVector tPoint;

    vtkIdType vMeshIdApexStart;
    vtkIdType vMeshIdApexEnd;
    deque<vtkIdType> vMeshIdGroup;
    deque<deque<vtkIdType>> vMeshIdSet;

    deque<deque<vtkIdType>>::iterator vThisGroup;
    deque<deque<vtkIdType>>::iterator vNextGroup;

    deque<vtkIdType>::iterator vThisThisPoint;
    deque<vtkIdType>::iterator vThisNextPoint;
    deque<vtkIdType>::iterator vNextThisPoint;
    deque<vtkIdType>::iterator vNextNextPoint;

    vtkSmartPointer<vtkTriangle> vTriangle;
    vtkSmartPointer<vtkQuad> vQuad;

    //create apex start point id
    LocalToGlobal(anApexStart, tPoint);
    vMeshIdApexStart = fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z());

    //create apex end point id
    LocalToGlobal(anApexEnd, tPoint);
    vMeshIdApexEnd = fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z());

    //create mesh point ids
    for (const auto& tSetIt : aMesh.fData) {
        vMeshIdGroup.clear();
        for (const auto& tGroupIt : tSetIt) {
            LocalToGlobal(tGroupIt, tPoint);
            vMeshIdGroup.push_back(fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z()));
        }
        vMeshIdGroup.push_back(vMeshIdGroup.front());
        vMeshIdSet.push_back(vMeshIdGroup);
    }

    //create start cap cells
    vThisThisPoint = vMeshIdSet.front().begin();
    vThisNextPoint = ++(vMeshIdSet.front().begin());
    while (vThisNextPoint != vMeshIdSet.front().end()) {
        vTriangle = vtkSmartPointer<vtkTriangle>::New();
        vTriangle->GetPointIds()->SetId(0, *vThisThisPoint);
        vTriangle->GetPointIds()->SetId(1, *vThisNextPoint);
        vTriangle->GetPointIds()->SetId(2, vMeshIdApexStart);
        fCells->InsertNextCell(vTriangle);
        fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                  fCurrentData->GetColor().GetGreen(),
                                  fCurrentData->GetColor().GetBlue(),
                                  fCurrentData->GetColor().GetOpacity());
        ++vThisThisPoint;
        ++vThisNextPoint;
    }

    //create hull cells
    vThisGroup = vMeshIdSet.begin();
    vNextGroup = ++(vMeshIdSet.begin());
    while (vNextGroup != vMeshIdSet.end()) {
        vThisThisPoint = vThisGroup->begin();
        vThisNextPoint = ++(vThisGroup->begin());
        vNextThisPoint = vNextGroup->begin();
        vNextNextPoint = ++(vNextGroup->begin());

        while (vNextNextPoint != vNextGroup->end()) {
            vQuad = vtkSmartPointer<vtkQuad>::New();
            vQuad->GetPointIds()->SetId(0, *vThisThisPoint);
            vQuad->GetPointIds()->SetId(1, *vNextThisPoint);
            vQuad->GetPointIds()->SetId(2, *vNextNextPoint);
            vQuad->GetPointIds()->SetId(3, *vThisNextPoint);
            fCells->InsertNextCell(vQuad);
            fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                      fCurrentData->GetColor().GetGreen(),
                                      fCurrentData->GetColor().GetBlue(),
                                      fCurrentData->GetColor().GetOpacity());
            ++vThisThisPoint;
            ++vThisNextPoint;
            ++vNextThisPoint;
            ++vNextNextPoint;
        }

        ++vThisGroup;
        ++vNextGroup;
    }

    //create end cap cells
    vThisThisPoint = vMeshIdSet.back().begin();
    vThisNextPoint = ++(vMeshIdSet.back().begin());
    while (vThisNextPoint != vMeshIdSet.back().end()) {
        vTriangle = vtkSmartPointer<vtkTriangle>::New();
        vTriangle->GetPointIds()->SetId(0, *vThisThisPoint);
        vTriangle->GetPointIds()->SetId(1, *vThisNextPoint);
        vTriangle->GetPointIds()->SetId(2, vMeshIdApexEnd);
        fCells->InsertNextCell(vTriangle);
        fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                  fCurrentData->GetColor().GetGreen(),
                                  fCurrentData->GetColor().GetBlue(),
                                  fCurrentData->GetColor().GetOpacity());
        ++vThisThisPoint;
        ++vThisNextPoint;
    }

    return;
}
void KGVTKGeometryPainter::ClosedShellMeshToVTK(const ShellMesh& aMesh)
{
    if (aMesh.fData.empty()) {
        vismsg(eWarning) << "mesh has size of zero, check your geometry" << eom;
    }

    //object allocation
    KThreeVector tPoint;

    deque<vtkIdType> vMeshIdGroup;
    deque<deque<vtkIdType>> vMeshIdSet;

    deque<deque<vtkIdType>>::iterator vThisGroup;
    deque<deque<vtkIdType>>::iterator vNextGroup;

    deque<vtkIdType>::iterator vThisThisPoint;
    deque<vtkIdType>::iterator vThisNextPoint;
    deque<vtkIdType>::iterator vNextThisPoint;
    deque<vtkIdType>::iterator vNextNextPoint;

    vtkSmartPointer<vtkQuad> vQuad;

    //create mesh point ids
    for (const auto& tSetIt : aMesh.fData) {
        vMeshIdGroup.clear();
        for (const auto& tGroupIt : tSetIt) {
            LocalToGlobal(tGroupIt, tPoint);
            vMeshIdGroup.push_back(fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z()));
        }

        vMeshIdSet.push_back(vMeshIdGroup);
    }
    vMeshIdSet.push_back(vMeshIdSet.front());

    //create hull cells
    vThisGroup = vMeshIdSet.begin();
    vNextGroup = ++(vMeshIdSet.begin());
    while (vNextGroup != vMeshIdSet.end()) {
        vThisThisPoint = vThisGroup->begin();
        vThisNextPoint = ++(vThisGroup->begin());
        vNextThisPoint = vNextGroup->begin();
        vNextNextPoint = ++(vNextGroup->begin());

        while (vNextNextPoint != vNextGroup->end()) {
            vQuad = vtkSmartPointer<vtkQuad>::New();
            vQuad->GetPointIds()->SetId(0, *vThisThisPoint);
            vQuad->GetPointIds()->SetId(1, *vNextThisPoint);
            vQuad->GetPointIds()->SetId(2, *vNextNextPoint);
            vQuad->GetPointIds()->SetId(3, *vThisNextPoint);
            fCells->InsertNextCell(vQuad);
            fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                      fCurrentData->GetColor().GetGreen(),
                                      fCurrentData->GetColor().GetBlue(),
                                      fCurrentData->GetColor().GetOpacity());
            ++vThisThisPoint;
            ++vThisNextPoint;
            ++vNextThisPoint;
            ++vNextNextPoint;
        }

        ++vThisGroup;
        ++vNextGroup;
    }

    return;
}
void KGVTKGeometryPainter::ShellMeshToVTK(const ShellMesh& aMesh)
{
    if (aMesh.fData.empty()) {
        vismsg(eWarning) << "mesh has size of zero, check your geometry" << eom;
    }

    //object allocation
    KThreeVector tPoint;

    deque<vtkIdType> vMeshIdGroup;
    deque<deque<vtkIdType>> vMeshIdSet;

    deque<deque<vtkIdType>>::iterator vThisGroup;
    deque<deque<vtkIdType>>::iterator vNextGroup;

    deque<vtkIdType>::iterator vThisThisPoint;
    deque<vtkIdType>::iterator vThisNextPoint;
    deque<vtkIdType>::iterator vNextThisPoint;
    deque<vtkIdType>::iterator vNextNextPoint;

    vtkSmartPointer<vtkQuad> vQuad;

    //create mesh point ids
    for (const auto& tSetIt : aMesh.fData) {
        vMeshIdGroup.clear();
        for (const auto& tGroupIt : tSetIt) {
            LocalToGlobal(tGroupIt, tPoint);
            vMeshIdGroup.push_back(fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z()));
        }

        vMeshIdSet.push_back(vMeshIdGroup);
    }

    //create hull cells
    vThisGroup = vMeshIdSet.begin();
    vNextGroup = ++(vMeshIdSet.begin());
    while (vNextGroup != vMeshIdSet.end()) {
        vThisThisPoint = vThisGroup->begin();
        vThisNextPoint = ++(vThisGroup->begin());
        vNextThisPoint = vNextGroup->begin();
        vNextNextPoint = ++(vNextGroup->begin());

        while (vNextNextPoint != vNextGroup->end()) {
            vQuad = vtkSmartPointer<vtkQuad>::New();
            vQuad->GetPointIds()->SetId(0, *vThisThisPoint);
            vQuad->GetPointIds()->SetId(1, *vNextThisPoint);
            vQuad->GetPointIds()->SetId(2, *vNextNextPoint);
            vQuad->GetPointIds()->SetId(3, *vThisNextPoint);
            fCells->InsertNextCell(vQuad);
            fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                      fCurrentData->GetColor().GetGreen(),
                                      fCurrentData->GetColor().GetBlue(),
                                      fCurrentData->GetColor().GetOpacity());
            ++vThisThisPoint;
            ++vThisNextPoint;
            ++vNextThisPoint;
            ++vNextNextPoint;
        }

        ++vThisGroup;
        ++vNextGroup;
    }

    return;
}
void KGVTKGeometryPainter::TorusMeshToVTK(const TorusMesh& aMesh)
{
    //object allocation
    KThreeVector tPoint;

    deque<vtkIdType> vMeshIdGroup;
    deque<deque<vtkIdType>> vMeshIdSet;

    deque<deque<vtkIdType>>::iterator vThisGroup;
    deque<deque<vtkIdType>>::iterator vNextGroup;

    deque<vtkIdType>::iterator vThisThisPoint;
    deque<vtkIdType>::iterator vThisNextPoint;
    deque<vtkIdType>::iterator vNextThisPoint;
    deque<vtkIdType>::iterator vNextNextPoint;

    vtkSmartPointer<vtkQuad> vQuad;

    //create mesh point ids
    for (const auto& tSetIt : aMesh.fData) {
        vMeshIdGroup.clear();
        for (const auto& tGroupIt : tSetIt) {
            LocalToGlobal(tGroupIt, tPoint);
            vMeshIdGroup.push_back(fPoints->InsertNextPoint(tPoint.X(), tPoint.Y(), tPoint.Z()));
        }
        vMeshIdGroup.push_back(vMeshIdGroup.front());
        vMeshIdSet.push_back(vMeshIdGroup);
    }
    vMeshIdSet.push_back(vMeshIdSet.front());

    //create hull cells
    vThisGroup = vMeshIdSet.begin();
    vNextGroup = ++(vMeshIdSet.begin());
    while (vNextGroup != vMeshIdSet.end()) {
        vThisThisPoint = vThisGroup->begin();
        vThisNextPoint = ++(vThisGroup->begin());
        vNextThisPoint = vNextGroup->begin();
        vNextNextPoint = ++(vNextGroup->begin());

        while (vNextNextPoint != vNextGroup->end()) {
            vQuad = vtkSmartPointer<vtkQuad>::New();
            vQuad->GetPointIds()->SetId(0, *vThisThisPoint);
            vQuad->GetPointIds()->SetId(1, *vNextThisPoint);
            vQuad->GetPointIds()->SetId(2, *vNextNextPoint);
            vQuad->GetPointIds()->SetId(3, *vThisNextPoint);
            fCells->InsertNextCell(vQuad);
            fColors->InsertNextTuple4(fCurrentData->GetColor().GetRed(),
                                      fCurrentData->GetColor().GetGreen(),
                                      fCurrentData->GetColor().GetBlue(),
                                      fCurrentData->GetColor().GetOpacity());
            ++vThisThisPoint;
            ++vThisNextPoint;
            ++vNextThisPoint;
            ++vNextNextPoint;
        }

        ++vThisGroup;
        ++vNextGroup;
    }

    return;
}

}  // namespace KGeoBag
