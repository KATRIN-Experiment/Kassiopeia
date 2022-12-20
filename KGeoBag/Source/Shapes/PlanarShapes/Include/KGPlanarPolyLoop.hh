#ifndef KGPLANARPOLYLOOP_HH_
#define KGPLANARPOLYLOOP_HH_

#include "KGPlanarArcSegment.hh"
#include "KGPlanarClosedPath.hh"
#include "KGPlanarLineSegment.hh"

#include <list>

namespace KGeoBag
{

class KGPlanarPolyLoop : public KGPlanarClosedPath
{
  public:
    typedef std::deque<const KGPlanarOpenPath*> Set;
    using It = Set::iterator;
    using CIt = Set::const_iterator;

  public:
    KGPlanarPolyLoop();
    KGPlanarPolyLoop(const KGPlanarPolyLoop& aCopy);
    ~KGPlanarPolyLoop() override;

    static std::string Name()
    {
        return "poly_loop";
    }

    KGPlanarPolyLoop* Clone() const override;
    void CopyFrom(const KGPlanarPolyLoop& aCopy);

  public:
    void StartPoint(const katrin::KTwoVector& aPoint);
    void NextLine(const katrin::KTwoVector& aVertex, const unsigned int aCount = 2, const double aPower = 1.);
    void NextArc(const katrin::KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong,
                 const unsigned int aCount = 2);
    void PreviousLine(const katrin::KTwoVector& aVertex, const unsigned int aCount = 2, const double aPower = 1.);
    void PreviousArc(const katrin::KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong,
                     const unsigned int aCount = 2);
    void LastLine(const unsigned int aCount = 2, const double aPower = 1.);
    void LastArc(const double& aRadius, const bool& aLeft, const bool& aLong, const unsigned int aCount = 2);

    const Set& Elements() const;

    const double& Length() const override;
    const katrin::KTwoVector& Centroid() const override;
    const katrin::KTwoVector& Anchor() const override;

  public:
    katrin::KTwoVector At(const double& aLength) const override;
    katrin::KTwoVector Point(const katrin::KTwoVector& aQuery) const override;
    katrin::KTwoVector Normal(const katrin::KTwoVector& aQuery) const override;
    bool Above(const katrin::KTwoVector& aQuery) const override;

  private:
    Set fElements;

    mutable double fLength;
    mutable katrin::KTwoVector fCentroid;
    mutable katrin::KTwoVector fAnchor;

    void Initialize() const;
    mutable bool fInitialized;

    mutable bool fIsCounterClockwise;

    //returns true if the loop runs counter-clockwise
    //returns false if the loop runs clockwise
    bool DetermineInteriorSide() const;

  public:
    class StartPointArguments
    {
      public:
        StartPointArguments() : fPoint(0., 0.) {}
        ~StartPointArguments() = default;

        katrin::KTwoVector fPoint;
    };

    class LineArguments
    {
      public:
        LineArguments() : fVertex(0., 0.), fMeshCount(1), fMeshPower(1.) {}
        ~LineArguments() = default;

        katrin::KTwoVector fVertex;
        unsigned int fMeshCount;
        double fMeshPower;
    };

    class ArcArguments
    {
      public:
        ArcArguments() : fVertex(0., 0.), fRadius(0.), fRight(true), fShort(true), fMeshCount(64) {}
        ~ArcArguments() = default;

        katrin::KTwoVector fVertex;
        double fRadius;
        bool fRight;
        bool fShort;
        unsigned int fMeshCount;
    };

    class LastLineArguments
    {
      public:
        LastLineArguments() : fMeshCount(1), fMeshPower(1.) {}
        ~LastLineArguments() = default;

        unsigned int fMeshCount;
        double fMeshPower;
    };

    class LastArcArguments
    {
      public:
        LastArcArguments() : fRadius(0.), fRight(true), fShort(true), fMeshCount(64) {}
        ~LastArcArguments() = default;

        double fRadius;
        bool fRight;
        bool fShort;
        unsigned int fMeshCount;
    };
};

}  // namespace KGeoBag

#endif
