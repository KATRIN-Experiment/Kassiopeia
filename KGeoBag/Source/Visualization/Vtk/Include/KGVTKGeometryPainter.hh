#ifndef KGVTKGEOMETRYPAINTER_HH_
#define KGVTKGEOMETRYPAINTER_HH_

#include "KGAppearance.hh"
#include "KGConicalWireArraySurface.hh"
#include "KGCore.hh"
#include "KGExtrudedArcSegmentSurface.hh"
#include "KGExtrudedCircleSpace.hh"
#include "KGExtrudedCircleSurface.hh"
#include "KGExtrudedLineSegmentSurface.hh"
#include "KGExtrudedPolyLineSurface.hh"
#include "KGExtrudedPolyLoopSpace.hh"
#include "KGExtrudedPolyLoopSurface.hh"
#include "KGFlattenedCircleSurface.hh"
#include "KGFlattenedPolyLoopSurface.hh"
#include "KGRodSpace.hh"
#include "KGRodSurface.hh"
#include "KGRotatedArcSegmentSpace.hh"
#include "KGRotatedArcSegmentSurface.hh"
#include "KGRotatedCircleSpace.hh"
#include "KGRotatedCircleSurface.hh"
#include "KGRotatedLineSegmentSpace.hh"
#include "KGRotatedLineSegmentSurface.hh"
#include "KGRotatedPolyLineSpace.hh"
#include "KGRotatedPolyLineSurface.hh"
#include "KGRotatedPolyLoopSpace.hh"
#include "KGRotatedPolyLoopSurface.hh"
#include "KGShellArcSegmentSurface.hh"
#include "KGShellCircleSurface.hh"
#include "KGShellLineSegmentSurface.hh"
#include "KGShellPolyLineSurface.hh"
#include "KGShellPolyLoopSurface.hh"
#include "KGStlFileSurface.hh"
#include "KVTKPainter.h"
#include "KVTKWindow.h"
#include "vtkActor.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkSmartPointer.h"

#include <deque>
#include <vector>
#include <string>

namespace KGeoBag
{

class KGVTKGeometryPainter :
    public katrin::KVTKPainter,
    public KGVisitor,
    public KGSurface::Visitor,
    public KGFlattenedCircleSurface::Visitor,
    public KGFlattenedPolyLoopSurface::Visitor,
    public KGRotatedLineSegmentSurface::Visitor,
    public KGRotatedArcSegmentSurface::Visitor,
    public KGRotatedPolyLineSurface::Visitor,
    public KGRotatedCircleSurface::Visitor,
    public KGRotatedPolyLoopSurface::Visitor,
    public KGShellLineSegmentSurface::Visitor,
    public KGShellArcSegmentSurface::Visitor,
    public KGShellPolyLineSurface::Visitor,
    public KGShellPolyLoopSurface::Visitor,
    public KGShellCircleSurface::Visitor,
    public KGExtrudedLineSegmentSurface::Visitor,
    public KGExtrudedArcSegmentSurface::Visitor,
    public KGExtrudedPolyLineSurface::Visitor,
    public KGExtrudedCircleSurface::Visitor,
    public KGExtrudedPolyLoopSurface::Visitor,
    public KGConicalWireArraySurface::Visitor,
    public KGRodSurface::Visitor,
    public KGStlFileSurface::Visitor,
    public KGSpace::Visitor,
    public KGRotatedLineSegmentSpace::Visitor,
    public KGRotatedArcSegmentSpace::Visitor,
    public KGRotatedPolyLineSpace::Visitor,
    public KGRotatedCircleSpace::Visitor,
    public KGRotatedPolyLoopSpace::Visitor,
    public KGExtrudedCircleSpace::Visitor,
    public KGExtrudedPolyLoopSpace::Visitor,
    public KGRodSpace::Visitor
{
  public:
    KGVTKGeometryPainter();
    ~KGVTKGeometryPainter() override;

  public:
    void Render() override;
    void Display() override;
    void Write() override;

  protected:
    void WriteVTK();
    void WriteSTL();

  public:
    void SetFile(const std::string& aName);
    const std::string& GetFile() const;
    void SetPath(const std::string& aPath);

    void SetWriteSTL(bool aFlag);

    void AddSurface(KGSurface* aSurface);
    void AddSpace(KGSpace* aSpace);

    std::string HelpText() override;
    void OnKeyPress(vtkObject* aCaller, long unsigned int eventId, void* aClient, void* callData) override;

  private:
    std::string fFile;
    std::string fPath;
    bool fWriteSTL;
    int fPlaneMode;

    std::vector<KGSurface*> fSurfaces;
    std::vector<KGSpace*> fSpaces;

    KGAppearanceData fDefaultData;

    //****************
    //surface visitors
    //****************

  protected:
    void VisitSurface(KGSurface* aSurface) override;
    void VisitFlattenedClosedPathSurface(KGFlattenedCircleSurface* aFlattenedCircleSurface) override;
    void VisitFlattenedClosedPathSurface(KGFlattenedPolyLoopSurface* aFlattenedPolyLoopSurface) override;
    void VisitRotatedPathSurface(KGRotatedLineSegmentSurface* aRotatedLineSegmentSurface) override;
    void VisitRotatedPathSurface(KGRotatedArcSegmentSurface* aRotatedArcSegmentSurface) override;
    void VisitRotatedPathSurface(KGRotatedPolyLineSurface* aRotatedPolyLineSurface) override;
    void VisitRotatedPathSurface(KGRotatedCircleSurface* aRotatedCircleSurface) override;
    void VisitRotatedPathSurface(KGRotatedPolyLoopSurface* aRotatedPolyLoopSurface) override;
    void VisitShellPathSurface(KGShellLineSegmentSurface* aShellLineSegmentSurface) override;
    void VisitShellPathSurface(KGShellArcSegmentSurface* aShellArcSegmentSurface) override;
    void VisitShellPathSurface(KGShellPolyLineSurface* aShellPolyLineSurface) override;
    void VisitShellPathSurface(KGShellPolyLoopSurface* aShellPolyLoopSurface) override;
    void VisitShellPathSurface(KGShellCircleSurface* aShellCircleSurface) override;
    void VisitExtrudedPathSurface(KGExtrudedLineSegmentSurface* aExtrudedLineSegmentSurface) override;
    void VisitExtrudedPathSurface(KGExtrudedArcSegmentSurface* aExtrudedArcSegmentSurface) override;
    void VisitExtrudedPathSurface(KGExtrudedPolyLineSurface* aExtrudedPolyLineSurface) override;
    void VisitExtrudedPathSurface(KGExtrudedCircleSurface* aExtrudedCircleSurface) override;
    void VisitExtrudedPathSurface(KGExtrudedPolyLoopSurface* aExtrudedPolyLoopSurface) override;
    void VisitWrappedSurface(KGConicalWireArraySurface* aConicalWireArraySurface) override;
    void VisitWrappedSurface(KGRodSurface* aRodSurface) override;
    void VisitWrappedSurface(KGStlFileSurface* aStlFileSurface) override;

    //**************
    //space visitors
    //**************

  protected:
    void VisitSpace(KGSpace* aSpace) override;
    void VisitRotatedOpenPathSpace(KGRotatedLineSegmentSpace* aRotatedLineSegmentSpace) override;
    void VisitRotatedOpenPathSpace(KGRotatedArcSegmentSpace* aRotatedArcSegmentSpace) override;
    void VisitRotatedOpenPathSpace(KGRotatedPolyLineSpace* aRotatedPolyLineSpace) override;
    void VisitRotatedClosedPathSpace(KGRotatedCircleSpace* aRotatedCircleSpace) override;
    void VisitRotatedClosedPathSpace(KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace) override;
    void VisitExtrudedClosedPathSpace(KGExtrudedCircleSpace* aExtrudedCircleSpace) override;
    void VisitExtrudedClosedPathSpace(KGExtrudedPolyLoopSpace* aExtrudedPolyLoopSpace) override;
    void VisitWrappedSpace(KGRodSpace* aRodSpace) override;

  private:
    void LocalToGlobal(const katrin::KThreeVector& aLocal, katrin::KThreeVector& aGlobal);

    //**********
    //data types
    //**********

    class Partition
    {
      public:
        typedef double Value;
        using Set = std::deque<Value>;
        using It = Set::iterator;
        using CIt = Set::const_iterator;

      public:
        Set fData;
    };

    class Points
    {
      public:
        using Element = katrin::KTwoVector;
        using Set = std::deque<Element>;
        using It = Set::iterator;
        using CIt = Set::const_iterator;

      public:
        Set fData;
    };

    class OpenPoints : public Points
    {};

    class ClosedPoints : public Points
    {};

    class ThreePoints
    {
      public:
        using Element = katrin::KThreeVector;
        using Set = std::deque<Element>;
        using It = Set::iterator;
        using CIt = Set::const_iterator;

      public:
        Set fData;
    };

    class Mesh
    {
      public:
        using Element = katrin::KThreeVector;
        using Group = std::deque<katrin::KThreeVector>;
        using GroupIt = Group::iterator;
        using GroupCIt = Group::const_iterator;
        using Set = std::deque<Group>;
        using SetIt = Set::iterator;
        using SetCIt = Set::const_iterator;

      public:
        Set fData;
    };

    class FlatMesh : public Mesh
    {};

    class TubeMesh : public Mesh
    {};

    class ShellMesh : public Mesh
    {};

    class TorusMesh : public Mesh
    {};

    //****************
    //points functions
    //****************

    static void LineSegmentToOpenPoints(const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints);
    void ArcSegmentToOpenPoints(const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints);
    void PolyLineToOpenPoints(const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints);
    void CircleToClosedPoints(const KGPlanarCircle* aCircle, ClosedPoints& aPoints);
    void PolyLoopToClosedPoints(const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints);
    static void RodsToThreePoints(const KGRodSpace* aRodSpace, ThreePoints& aThreePoints);
    static void RodsToThreePoints(const KGRodSurface* aRodSurface, ThreePoints& aThreePoints);
    static void WireArrayToThreePoints(const KGConicalWireArraySurface* aConicalWireArraySurface,
                                       ThreePoints& aThreePoints);

    //**************
    //mesh functions
    //**************

    static void ClosedPointsFlattenedToTubeMeshAndApex(const ClosedPoints& aPoints, const katrin::KTwoVector& aCentroid,
                                                       const double& aZ, TubeMesh& aMesh,
                                                       katrin::KThreeVector& anApex);
    void OpenPointsRotatedToTubeMesh(const OpenPoints& aPoints, TubeMesh& aMesh);
    void OpenPointsRotatedToShellMesh(const OpenPoints& aPoints, ShellMesh& aMesh, const double& aAngleStart,
                                      const double& aAngleStop);
    void ClosedPointsRotatedToShellMesh(const ClosedPoints& aPoints, ShellMesh& aMesh, const double& aAngleStart,
                                        const double& aAngleStop);
    void ClosedPointsRotatedToTorusMesh(const ClosedPoints& aPoints, TorusMesh& aMesh);
    static void OpenPointsExtrudedToFlatMesh(const OpenPoints& aPoints, const double& aZMin, const double& aZMax,
                                             FlatMesh& aMesh);
    static void ClosedPointsExtrudedToTubeMesh(const ClosedPoints& aPoints, const double& aZMin, const double& aZMax,
                                               TubeMesh& aMesh);
    void ThreePointsToTubeMesh(const ThreePoints& aThreePoints, TubeMesh& aMesh, const double& aTubeRadius);

    //****************************
    //mesh and rendering functions
    //****************************

    void ThreePointsToTubeMeshToVTK(const ThreePoints& aThreePoints, TubeMesh& aMesh, const double& aTubeRadius);

    //*******************
    //rendering functions
    //*******************

    void MeshToVTK(const Mesh& aMesh);
    void FlatMeshToVTK(const FlatMesh& aMesh);
    void TubeMeshToVTK(const TubeMesh& aMesh);
    void TubeMeshToVTK(const TubeMesh& aMesh, const katrin::KThreeVector& anApexEnd);
    void TubeMeshToVTK(const katrin::KThreeVector& anApexStart, const TubeMesh& aMesh);
    void TubeMeshToVTK(const katrin::KThreeVector& anApexStart, const TubeMesh& aMesh,
                       const katrin::KThreeVector& anApexEnd);
    void ShellMeshToVTK(const ShellMesh& aMesh);
    void ClosedShellMeshToVTK(const ShellMesh& aMesh);
    void TorusMeshToVTK(const TorusMesh& aMesh);

  private:
    vtkSmartPointer<vtkPoints> fPoints;
    vtkSmartPointer<vtkCellArray> fCells;
    vtkSmartPointer<vtkUnsignedCharArray> fColors;
    vtkSmartPointer<vtkPolyData> fPolyData;
    vtkSmartPointer<vtkPolyDataMapper> fMapper;
    vtkSmartPointer<vtkActor> fActor;

    KGSpace* fCurrentSpace;
    KGSurface* fCurrentSurface;
    KGAppearanceData* fCurrentData;
    katrin::KThreeVector fCurrentOrigin;
    katrin::KThreeVector fCurrentXAxis;
    katrin::KThreeVector fCurrentYAxis;
    katrin::KThreeVector fCurrentZAxis;
    bool fIgnore;
};

}  // namespace KGeoBag

#endif
