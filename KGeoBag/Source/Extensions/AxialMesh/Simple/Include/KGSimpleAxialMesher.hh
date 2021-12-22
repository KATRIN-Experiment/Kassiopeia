#ifndef KGeoBag_KGSimpleAxialMesher_hh_
#define KGeoBag_KGSimpleAxialMesher_hh_

#include "KGAxialMesherBase.hh"
#include "KGPlanarArcSegment.hh"
#include "KGPlanarCircle.hh"
#include "KGPlanarLineSegment.hh"
#include "KGPlanarPolyLine.hh"
#include "KGPlanarPolyLoop.hh"

#include <deque>

namespace KGeoBag
{

class KGSimpleAxialMesher : virtual public KGAxialMesherBase
{
  public:
    KGSimpleAxialMesher();
    ~KGSimpleAxialMesher() override;

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
    void EndToOpenPoints(const katrin::KTwoVector& anEnd, const unsigned int& aMeshCount, const double& aMeshPower,
                         OpenPoints& aPoints);
    void LineSegmentToOpenPoints(const KGPlanarLineSegment* aLineSegment, OpenPoints& aPoints);
    void ArcSegmentToOpenPoints(const KGPlanarArcSegment* anArcSegment, OpenPoints& aPoints);
    void PolyLineToOpenPoints(const KGPlanarPolyLine* aPolyLine, OpenPoints& aPoints);
    void CircleToClosedPoints(const KGPlanarCircle* aCircle, ClosedPoints& aPoints);
    void PolyLoopToClosedPoints(const KGPlanarPolyLoop* aPolyLoop, ClosedPoints& aPoints);

    //*********************
    //tesselation functions
    //*********************

  protected:
    void OpenPointsToLoops(const OpenPoints& aMesh);
    void ClosedPointsToLoops(const ClosedPoints& aMesh);

    //*************
    //loop function
    //*************

  protected:
    void Loop(const katrin::KTwoVector& aFirst, const katrin::KTwoVector& aSecond);
};

}  // namespace KGeoBag

#endif
