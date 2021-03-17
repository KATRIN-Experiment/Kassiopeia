#ifndef KGeoBag_KGSimpleMesher_hh_
#define KGeoBag_KGSimpleMesher_hh_

#include "KGMesherBase.hh"
#include "KGPlanarArcSegment.hh"
#include "KGPlanarCircle.hh"
#include "KGPlanarLineSegment.hh"
#include "KGPlanarPolyLine.hh"
#include "KGPlanarPolyLoop.hh"

#include <deque>

namespace KGeoBag
{

class KGSimpleMesher : virtual public KGMesherBase
{
  public:
    using KGMesherBase::VisitExtendedSpace;
    using KGMesherBase::VisitExtendedSurface;

  public:
    KGSimpleMesher();
    ~KGSimpleMesher() override;

    //**********
    //data types
    //**********

  protected:
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
        using Element = KTwoVector;
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

    class Mesh
    {
      public:
        using Element = KGeoBag::KThreeVector;
        using Group = std::deque<KThreeVector>;
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

    //*******************
    //partition functions
    //*******************

  protected:
    static void SymmetricPartition(const double& aStart, const double& aStop, const unsigned int& aCount,
                                   const double& aPower, Partition& aPartition);
    static void ForwardPartition(const double& aStart, const double& aStop, const unsigned int& aCount,
                                 const double& aPower, Partition& aPartition);
    static void BackwardPartition(const double& aStart, const double& aStop, const unsigned int& aCount,
                                  const double& aPower, Partition& aPartition);

    //****************
    //points functions
    //****************

  protected:
    void LineSegmentToOpenPoints(const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints);
    void ArcSegmentToOpenPoints(const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints);
    void PolyLineToOpenPoints(const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints);
    void CircleToClosedPoints(const KGPlanarCircle* aCircle, ClosedPoints& aPoints);
    void PolyLoopToClosedPoints(const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints);

    //**************
    //mesh functions
    //**************

  protected:
    void ClosedPointsFlattenedToTubeMeshAndApex(const ClosedPoints& aPoints, const KTwoVector& aCentroid,
                                                const double& aZ, const unsigned int& aCount, const double& aPower,
                                                TubeMesh& aMesh, KGeoBag::KThreeVector& anApex);
    void OpenPointsRotatedToTubeMesh(const OpenPoints& aPoints, const unsigned int& aCount, TubeMesh& aMesh);
    void OpenPointsRotatedToShellMesh(const OpenPoints& aPoints, const unsigned int& aCount, const double& aPower,
                                      ShellMesh& aMesh, const double& aAngleStart, const double& aAngleStop);
    void ClosedPointsRotatedToShellMesh(const ClosedPoints& aPoints, const unsigned int& aCount, const double& aPower,
                                        ShellMesh& aMesh, const double& aAngleStart, const double& aAngleStop);
    void ClosedPointsRotatedToTorusMesh(const ClosedPoints& aPoints, const unsigned int& aCount, TorusMesh& aMesh);
    void OpenPointsExtrudedToFlatMesh(const OpenPoints& aPoints, const double& aZMin, const double& aZMax,
                                      const unsigned int& aCount, const double& aPower, FlatMesh& aMesh);
    void ClosedPointsExtrudedToTubeMesh(const ClosedPoints& aPoints, const double& aZMin, const double& aZMax,
                                        const unsigned int& aCount, const double& aPower, TubeMesh& aMesh);

    //*********************
    //tesselation functions
    //*********************

  protected:
    void FlatMeshToTriangles(const FlatMesh& aMesh);
    void TubeMeshToTriangles(const TubeMesh& aMesh);
    void TubeMeshToTriangles(const TubeMesh& aMesh, const KGeoBag::KThreeVector& anApexEnd);
    void TubeMeshToTriangles(const KGeoBag::KThreeVector& anApexStart, const TubeMesh& aMesh);
    void TubeMeshToTriangles(const KGeoBag::KThreeVector& anApexStart, const TubeMesh& aMesh,
                             const KGeoBag::KThreeVector& anApexEnd);
    void ShellMeshToTriangles(const ShellMesh& aMesh);
    void ClosedShellMeshToTriangles(const ShellMesh& aMesh);
    void TorusMeshToTriangles(const TorusMesh& aMesh);

    //*****************
    //triangle function
    //*****************

  protected:
    void Triangle(const KGeoBag::KThreeVector& aFirst, const KGeoBag::KThreeVector& aSecond,
                  const KGeoBag::KThreeVector& aThird);
};

}  // namespace KGeoBag

#endif
