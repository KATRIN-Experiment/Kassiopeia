#ifndef KGVTKGEOMETRYPAINTER_HH_
#define KGVTKGEOMETRYPAINTER_HH_

#include "KVTKWindow.h"
using katrin::KVTKWindow;

#include "KVTKPainter.h"
using katrin::KVTKPainter;

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
#include "vtkActor.h"
#include "vtkCellArray.h"
#include "vtkCellData.h"
#include "vtkPoints.h"
#include "vtkPolyData.h"
#include "vtkPolyDataMapper.h"
#include "vtkSmartPointer.h"

#include <deque>
#include <vector>

namespace KGeoBag
{

class KGVTKGeometryPainter :
    public KVTKPainter,
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
    virtual ~KGVTKGeometryPainter();

  public:
    void Render();
    void Display();
    void Write();

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

  private:
    std::string fFile;
    std::string fPath;
    bool fWriteSTL;

    std::vector<KGSurface*> fSurfaces;
    std::vector<KGSpace*> fSpaces;

    KGAppearanceData fDefaultData;

    //****************
    //surface visitors
    //****************

  protected:
    virtual void VisitSurface(KGSurface* aSurface);
    virtual void VisitFlattenedClosedPathSurface(KGFlattenedCircleSurface* aFlattenedCircleSurface);
    virtual void VisitFlattenedClosedPathSurface(KGFlattenedPolyLoopSurface* aFlattenedPolyLoopSurface);
    virtual void VisitRotatedPathSurface(KGRotatedLineSegmentSurface* aRotatedLineSegmentSurface);
    virtual void VisitRotatedPathSurface(KGRotatedArcSegmentSurface* aRotatedArcSegmentSurface);
    virtual void VisitRotatedPathSurface(KGRotatedPolyLineSurface* aRotatedPolyLineSurface);
    virtual void VisitRotatedPathSurface(KGRotatedCircleSurface* aRotatedCircleSurface);
    virtual void VisitRotatedPathSurface(KGRotatedPolyLoopSurface* aRotatedPolyLoopSurface);
    virtual void VisitShellPathSurface(KGShellLineSegmentSurface* aShellLineSegmentSurface);
    virtual void VisitShellPathSurface(KGShellArcSegmentSurface* aShellArcSegmentSurface);
    virtual void VisitShellPathSurface(KGShellPolyLineSurface* aShellPolyLineSurface);
    virtual void VisitShellPathSurface(KGShellPolyLoopSurface* aShellPolyLoopSurface);
    virtual void VisitShellPathSurface(KGShellCircleSurface* aShellCircleSurface);
    virtual void VisitExtrudedPathSurface(KGExtrudedLineSegmentSurface* aExtrudedLineSegmentSurface);
    virtual void VisitExtrudedPathSurface(KGExtrudedArcSegmentSurface* aExtrudedArcSegmentSurface);
    virtual void VisitExtrudedPathSurface(KGExtrudedPolyLineSurface* aExtrudedPolyLineSurface);
    virtual void VisitExtrudedPathSurface(KGExtrudedCircleSurface* aExtrudedCircleSurface);
    virtual void VisitExtrudedPathSurface(KGExtrudedPolyLoopSurface* aExtrudedPolyLoopSurface);
    virtual void VisitWrappedSurface(KGConicalWireArraySurface* aConicalWireArraySurface);
    virtual void VisitWrappedSurface(KGRodSurface* aRodSurface);

    //**************
    //space visitors
    //**************

  protected:
    virtual void VisitSpace(KGSpace* aSpace);
    virtual void VisitRotatedOpenPathSpace(KGRotatedLineSegmentSpace* aRotatedLineSegmentSpace);
    virtual void VisitRotatedOpenPathSpace(KGRotatedArcSegmentSpace* aRotatedArcSegmentSpace);
    virtual void VisitRotatedOpenPathSpace(KGRotatedPolyLineSpace* aRotatedPolyLineSpace);
    virtual void VisitRotatedClosedPathSpace(KGRotatedCircleSpace* aRotatedCircleSpace);
    virtual void VisitRotatedClosedPathSpace(KGRotatedPolyLoopSpace* aRotatedPolyLoopSpace);
    virtual void VisitExtrudedClosedPathSpace(KGExtrudedCircleSpace* aExtrudedCircleSpace);
    virtual void VisitExtrudedClosedPathSpace(KGExtrudedPolyLoopSpace* aExtrudedPolyLoopSpace);
    virtual void VisitWrappedSpace(KGRodSpace* aRodSpace);

  private:
    void LocalToGlobal(const KThreeVector& aLocal, KThreeVector& aGlobal);

    //**********
    //data types
    //**********

    class Partition
    {
      public:
        typedef double Value;
        typedef deque<Value> Set;
        typedef Set::iterator It;
        typedef Set::const_iterator CIt;

      public:
        Set fData;
    };

    class Points
    {
      public:
        typedef KTwoVector Element;
        typedef deque<Element> Set;
        typedef Set::iterator It;
        typedef Set::const_iterator CIt;

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
        typedef KThreeVector Element;
        typedef deque<Element> Set;
        typedef Set::iterator It;
        typedef Set::const_iterator CIt;

      public:
        Set fData;
    };

    class Mesh
    {
      public:
        typedef KThreeVector Element;
        typedef deque<KThreeVector> Group;
        typedef Group::iterator GroupIt;
        typedef Group::const_iterator GroupCIt;
        typedef deque<Group> Set;
        typedef Set::iterator SetIt;
        typedef Set::const_iterator SetCIt;

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

    void LineSegmentToOpenPoints(const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints);
    void ArcSegmentToOpenPoints(const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints);
    void PolyLineToOpenPoints(const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints);
    void CircleToClosedPoints(const KGPlanarCircle* aCircle, ClosedPoints& aPoints);
    void PolyLoopToClosedPoints(const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints);
    void RodsToThreePoints(const KGRodSpace* aRodSpace, ThreePoints& aThreePoints);
    void RodsToThreePoints(const KGRodSurface* aRodSurface, ThreePoints& aThreePoints);
    void WireArrayToThreePoints(const KGConicalWireArraySurface* aConicalWireArraySurface, ThreePoints& aThreePoints);

    //**************
    //mesh functions
    //**************

    void ClosedPointsFlattenedToTubeMeshAndApex(const ClosedPoints& aPoints, const KTwoVector& aCentroid,
                                                const double& aZ, TubeMesh& aMesh, KThreeVector& anApex);
    void OpenPointsRotatedToTubeMesh(const OpenPoints& aPoints, TubeMesh& aMesh);
    void OpenPointsRotatedToShellMesh(const OpenPoints& aPoints, ShellMesh& aMesh, const double& aAngleStart,
                                      const double& aAngleStop);
    void ClosedPointsRotatedToShellMesh(const ClosedPoints& aPoints, ShellMesh& aMesh, const double& aAngleStart,
                                        const double& aAngleStop);
    void ClosedPointsRotatedToTorusMesh(const ClosedPoints& aPoints, TorusMesh& aMesh);
    void OpenPointsExtrudedToFlatMesh(const OpenPoints& aPoints, const double& aZMin, const double& aZMax,
                                      FlatMesh& aMesh);
    void ClosedPointsExtrudedToTubeMesh(const ClosedPoints& aPoints, const double& aZMin, const double& aZMax,
                                        TubeMesh& aMesh);
    void ThreePointsToTubeMesh(const ThreePoints& aThreePoints, TubeMesh& aMesh, const double& aTubeRadius);

    //****************************
    //mesh and rendering functions
    //****************************

    void ThreePointsToTubeMeshToVTK(const ThreePoints& aThreePoints, TubeMesh& aMesh, const double& aTubeRadius);

    //*******************
    //rendering functions
    //*******************

    void FlatMeshToVTK(const FlatMesh& aMesh);
    void TubeMeshToVTK(const TubeMesh& aMesh);
    void TubeMeshToVTK(const TubeMesh& aMesh, const KThreeVector& anApexEnd);
    void TubeMeshToVTK(const KThreeVector& anApexStart, const TubeMesh& aMesh);
    void TubeMeshToVTK(const KThreeVector& anApexStart, const TubeMesh& aMesh, const KThreeVector& anApexEnd);
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
    KThreeVector fCurrentOrigin;
    KThreeVector fCurrentXAxis;
    KThreeVector fCurrentYAxis;
    KThreeVector fCurrentZAxis;
    bool fIgnore;
};

}  // namespace KGeoBag

#endif
