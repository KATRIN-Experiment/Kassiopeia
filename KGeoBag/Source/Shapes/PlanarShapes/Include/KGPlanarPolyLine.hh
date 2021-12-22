#ifndef KGPLANARPOLYLINE_HH_
#define KGPLANARPOLYLINE_HH_

#include "KGPlanarArcSegment.hh"
#include "KGPlanarLineSegment.hh"
#include "KGPlanarOpenPath.hh"

namespace KGeoBag
{

class KGPlanarPolyLine : public KGPlanarOpenPath
{
  public:
    typedef std::deque<const KGPlanarOpenPath*> Set;
    using It = Set::iterator;
    using CIt = Set::const_iterator;

  public:
    KGPlanarPolyLine();
    KGPlanarPolyLine(const KGPlanarPolyLine& aCopy);
    ~KGPlanarPolyLine() override;

    static std::string Name()
    {
        return "poly_line";
    }

    KGPlanarPolyLine* Clone() const override;
    void CopyFrom(const KGPlanarPolyLine& aCopy);

  public:
    void StartPoint(const katrin::KTwoVector& aPoint);
    void NextLine(const katrin::KTwoVector& aVertex, const unsigned int aCount = 2, const double aPower = 1.);
    void NextArc(const katrin::KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong,
                 const unsigned int aCount = 2);
    void PreviousLine(const katrin::KTwoVector& aVertex, const unsigned int aCount = 2, const double aPower = 1.);
    void PreviousArc(const katrin::KTwoVector& aVertex, const double& aRadius, const bool& aLeft, const bool& aLong,
                     const unsigned int aCount = 2);

    const Set& Elements() const;

    const double& Length() const override;
    const katrin::KTwoVector& Centroid() const override;
    const katrin::KTwoVector& Start() const override;
    const katrin::KTwoVector& End() const override;

  public:
    katrin::KTwoVector At(const double& aLength) const override;
    katrin::KTwoVector Point(const katrin::KTwoVector& aQuery) const override;
    katrin::KTwoVector Normal(const katrin::KTwoVector& aQuery) const override;
    bool Above(const katrin::KTwoVector& aQuery) const override;

  private:
    Set fElements;

    mutable double fLength;
    mutable katrin::KTwoVector fCentroid;
    mutable katrin::KTwoVector fStart;
    mutable katrin::KTwoVector fEnd;

    void Initialize() const;
    mutable bool fInitialized;

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
};

}  // namespace KGeoBag

#endif
