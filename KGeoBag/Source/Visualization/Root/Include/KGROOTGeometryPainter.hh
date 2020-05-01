#ifndef KGROOTGEOMETRYPAINTER_HH_
#define KGROOTGEOMETRYPAINTER_HH_

#include "KROOTWindow.h"
using katrin::KROOTWindow;

#include "KROOTPainter.h"
using katrin::KROOTPainter;

#include "KGAppearance.hh"
#include "KGBeam.hh"
#include "KGBeamSurface.hh"
#include "KGComplexAnnulus.hh"
#include "KGComplexAnnulusSurface.hh"
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
#include "KGPortHousingSurface.hh"
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
#include "KGWrappedSurface.hh"


//include root stuff
#include "TPolyLine.h"

#include <vector>
using std::vector;

#include <deque>
using std::deque;

#include <list>
using std::list;

#include <utility>
using std::pair;

#include <algorithm>
using std::reverse;
using std::swap;

#include "KField.h"

namespace KGeoBag
{

class KGROOTGeometryPainter :
    public KROOTPainter,
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
    public KGPortHousingSurface::Visitor,
    public KGBeamSurface::Visitor,
    public KGComplexAnnulusSurface::Visitor,
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
    KGROOTGeometryPainter();
    ~KGROOTGeometryPainter() override;

  public:
    void Render() override;
    void Display() override;
    void Write() override;

    void AddSurface(KGSurface* aSurface);
    void AddSpace(KGSpace* aSpace);

    double GetXMin() override;
    double GetXMax() override;
    double GetYMin() override;
    double GetYMax() override;

  private:
    vector<KGSurface*> fSurfaces;
    vector<KGSpace*> fSpaces;

    KGAppearanceData fDefaultData;

    //****************
    //Plane settings
    //****************

    ;
    K_SET(KThreeVector, PlaneNormal);
    K_SET(KThreeVector, PlanePoint);
    K_SET(bool, SwapAxis);
    K_GET(KThreeVector, PlaneVectorA);
    K_GET(KThreeVector, PlaneVectorB)

  public:
    std::string GetXAxisLabel() override;
    std::string GetYAxisLabel() override;

  private:
    std::string GetAxisLabel(KThreeVector anAxis);

  public:
    void CalculatePlaneCoordinateSystem();

    K_SET(double, Epsilon);


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
    void VisitWrappedSurface(KGPortHousingSurface* aPortHousingSurface) override;
    void VisitWrappedSurface(KGBeamSurface* aBeamSurface) override;
    void VisitWrappedSurface(KGComplexAnnulusSurface* aComplexAnnulus) override;
    void VisitWrappedSurface(KGRodSurface* aRodSurface) override;

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
    void LocalToGlobal(const KThreeVector& aLocal, KThreeVector& aGlobal);
    double distance(KTwoVector Vector1, KTwoVector Vector2);

    //**********
    //data types
    //**********

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

    class Mesh
    {
      public:
        typedef KThreeVector Element;
        typedef deque<Element> Group;
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

    class TorusMesh : public Mesh
    {};

    class ShellMesh : public Mesh
    {};

    class PortMesh : public Mesh
    {};
    class BeamMesh : public Mesh
    {};
    class RingMesh : public Mesh
    {};


    class Lines
    {
      public:
        typedef KThreeVector Element;
        typedef pair<Element, Element> Line;
        typedef deque<Line> Group;
        typedef Group::iterator GroupIt;
        typedef Group::const_iterator GroupCIt;
        typedef deque<Group> Set;
        typedef Set::iterator SetIt;
        typedef Set::const_iterator SetCIt;

      public:
        Set fData;
    };

    class CircleLines : public Lines
    {};

    class ParallelLines : public Lines
    {};

    class ArcLines : public Lines
    {};


    class IntersectionPoints
    {
      public:
        typedef KTwoVector Element;
        typedef deque<Element> Group;
        typedef Group::iterator GroupIt;
        typedef Group::const_iterator GroupCIt;
        typedef enum
        {
            eUndefined,
            eParallel,
            eCircle
        } Origin;
        typedef pair<Group, Origin> NamedGroup;
        typedef deque<NamedGroup> Set;
        typedef Set::iterator SetIt;
        typedef Set::const_iterator SetCIt;

      public:
        Set fData;
    };

    class OrderedPoints
    {
      public:
        typedef Points Element;
        typedef deque<Element> Set;
        typedef Set::iterator SetIt;
        typedef Set::const_iterator SetCIt;

      public:
        Set fData;
    };

    class SubPortOrderedPoints
    {
      public:
        typedef OrderedPoints Element;
        typedef deque<Element> Set;
        typedef Set::iterator SetIt;
        typedef Set::const_iterator SetCIt;

      public:
        Set fData;
    };

    class ConnectionPoints
    {
      public:
        typedef pair<KTwoVector, OrderedPoints::SetCIt> Element;
        typedef deque<Element> Group;
        typedef Group::iterator GroupIt;
        typedef Group::const_iterator GroupCIt;
        typedef deque<Group> Set;
        typedef Set::iterator SetIt;
        typedef Set::const_iterator SetCIt;

      public:
        Set fData;
    };


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


    //****************
    //points functions
    //****************

    void LineSegmentToOpenPoints(const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints);
    void ArcSegmentToOpenPoints(const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints);
    void PolyLineToOpenPoints(const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints);
    void CircleToClosedPoints(const KGPlanarCircle* aCircle, ClosedPoints& aPoints);
    void PolyLoopToClosedPoints(const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints);
    void RodToOpenPoints(const KGRod* aRod, OpenPoints& aPoints);

    //**************
    //mesh functions
    //**************

    void ClosedPointsFlattenedToTubeMeshAndApex(const ClosedPoints& aPoints, const KTwoVector& aCentroid,
                                                const double& aZ, TubeMesh& aMesh, KThreeVector& anApex);
    void OpenPointsRotatedToTubeMesh(const OpenPoints& aPoints, TubeMesh& aMesh);
    void ClosedPointsRotatedToTorusMesh(const ClosedPoints& aPoints, TorusMesh& aMesh);
    void OpenPointsExtrudedToFlatMesh(const OpenPoints& aPoints, const double& aZMin, const double& aZMax,
                                      FlatMesh& aMesh);
    void ClosedPointsExtrudedToTubeMesh(const ClosedPoints& aPoints, const double& aZMin, const double& aZMax,
                                        TubeMesh& aMesh);
    void OpenPointsToShellMesh(const OpenPoints& aPoints, ShellMesh& aMesh, const unsigned int& aCount,
                               const double& aPower, const double& AngleStart, const double& AngleStop);
    void ClosedPointsToMainPortMesh(const double* PointA, const double* PointB, const double aRadius, PortMesh& aMesh);
    void ClosedPointsToSubPortMesh(const KGPortHousing::CircularPort* aCircularPort, PortMesh& aMesh);
    void ClosedPointsToBeamMesh(const vector<vector<double>> aStartCoord, const vector<vector<double>> aEndCoord,
                                BeamMesh& aMesh);
    void ClosedPointsToFlatMesh(const std::shared_ptr<KGComplexAnnulus> aComplexAnnulus, FlatMesh& aMesh);
    void ClosedPointsToRingMesh(const std::shared_ptr<KGComplexAnnulus> aComplexAnnulus, RingMesh& aMesh);


    //**************
    //line functions
    //**************

    void ShellMeshToArcLines(const ShellMesh aMesh, ArcLines& anArcLines);
    void ShellMeshToParallelLines(const ShellMesh aMesh, ParallelLines& aParallelLines);
    void TubeMeshToCircleLines(const TubeMesh aMesh, CircleLines& aCircleLines);
    void TubeMeshToParallelLines(const TubeMesh aMesh, ParallelLines& aParallelLines);
    void TorusMeshToCircleLines(const TorusMesh aMesh, CircleLines& aCircleLines);
    void TorusMeshToParallelLines(const TorusMesh aMesh, ParallelLines& aParallelLines);
    void PortMeshToCircleLines(const PortMesh aMesh, CircleLines& aCircleLines);
    void PortMeshToParallelLines(const PortMesh aMesh, ParallelLines& aParallelLines);
    void BeamMeshToCircleLines(const BeamMesh aMesh, CircleLines& aCircleLines);
    void BeamMeshToParallelLines(const BeamMesh aMesh, ParallelLines& aParallelLines);
    void FlatMeshToCircleLines(const FlatMesh aMesh, CircleLines& aCircleLines);
    void RingMeshToCircleLines(const RingMesh aMesh, CircleLines& aCicleLines);


    //**********************
    //intersection functions
    //**********************

    void LinesToIntersections(const CircleLines aCircleLinesSet, IntersectionPoints& anIntersectionPoints);
    void LinesToIntersections(const ArcLines aCircleLinesSet, IntersectionPoints& anIntersectionPoints);
    void LinesToIntersections(const ParallelLines aCircleLinesSet, IntersectionPoints& anIntersectionPoints);
    void CalculatePlaneIntersection(const KThreeVector aStartPoint, const KThreeVector anEndPoint,
                                    KThreeVector& anIntersectionPoint, bool& anIntersection);
    void TransformToPlaneSystem(const KThreeVector aPoint, KTwoVector& aPlanePoint);

    void IntersectionPointsToOrderedPoints(const IntersectionPoints anIntersectionPoints,
                                           OrderedPoints& anOrderedPoints);
    void IntersectionPointsToOrderedPoints(const IntersectionPoints aMainIntersectionPoints,
                                           const IntersectionPoints aRingIntersectionPoints,
                                           OrderedPoints& anOrderdPoints);
    void ShellIntersectionPointsToOrderedPoints(const IntersectionPoints anIntersectionPoints,
                                                OrderedPoints& OrderedPoints);

    void CreateClosedOrderedPoints(const IntersectionPoints anIntersectionPoints, OrderedPoints& anOrderedPoints);
    void CreateShellClosedOrderedPoints(const IntersectionPoints anIntersectionPoints, OrderedPoints& anOrderedPoints);
    void CreateOpenOrderedPoints(const IntersectionPoints anIntersectionPoints, OrderedPoints& anOrderedPoints);
    void CreateShellOpenOrderedPoints(const IntersectionPoints anIntersectionPoints, OrderedPoints& anOrderedPoints);
    void CreateDualOrderedPoints(const IntersectionPoints anIntersectionPoints, OrderedPoints& anOrderedPoints);

    void CombineOrderedPoints(OrderedPoints& anOrderedPoints);
    void CombineOrderedPoints(OrderedPoints& aMainOrderedPoints, SubPortOrderedPoints& aSubOrderedPoints,
                              OrderedPoints& anOrderedPoints);

    //*******************
    //partition functions
    //*******************

    void SymmetricPartition(const double& aStart, const double& aStop, const unsigned int& aCount, const double& aPower,
                            Partition& aPartition);


    //*******************
    //rendering functions
    //*******************

    void OrderedPointsToROOTSurface(const OrderedPoints anOrderedPoints);
    void OrderedPointsToROOTSpace(const OrderedPoints anOrderedPoints);


  private:
    //root stuff
    vector<TPolyLine*> fROOTSpaces;
    vector<TPolyLine*> fROOTSurfaces;

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
